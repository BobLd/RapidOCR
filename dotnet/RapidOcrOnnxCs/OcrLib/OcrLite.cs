﻿using Emgu.CV;
using Emgu.CV.CvEnum;
using System;
using System.Collections.Generic;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using SkiaSharp;

namespace OcrLiteLib
{
    public class OcrLite
    {
        public bool isPartImg { get; set; }
        public bool isDebugImg { get; set; }
        private DbNet2 dbNet;
        private AngleNet angleNet;
        private CrnnNet crnnNet;

        public OcrLite()
        {
            dbNet = new DbNet2();
            angleNet = new AngleNet();
            crnnNet = new CrnnNet();
        }

        public void InitModels(string detPath, string clsPath, string recPath, string keysPath, int numThread)
        {
            try
            {
                dbNet.InitModel(detPath, numThread);
                angleNet.InitModel(clsPath, numThread);
                crnnNet.InitModel(recPath, keysPath, numThread);
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message + ex.StackTrace);
                throw ex;
            }
        }

        public OcrResult Detect(string img, int padding, int maxSideLen, float boxScoreThresh, float boxThresh,
                              float unClipRatio, bool doAngle, bool mostAngle)
        {
            SKData data = SKData.Create(img);
            var bounds = SKBitmap.DecodeBounds(data);
            bounds.ColorType = SKColorType.Rgb888x;
            bounds.AlphaType = SKAlphaType.Opaque;

            SKBitmap originSrc = new SKBitmap(bounds); // SKBitmap.Decode(img);
            originSrc.SetPixels(data.Handle);

            originSrc = SKBitmap.Decode(img);

            int originMaxSide = Math.Max(originSrc.Width, originSrc.Height);

            int resize;
            if (maxSideLen <= 0 || maxSideLen > originMaxSide)
            {
                resize = originMaxSide;
            }
            else
            {
                resize = maxSideLen;
            }
            resize += 2 * padding;
            Rectangle paddingRect = new Rectangle(padding, padding, originSrc.Width, originSrc.Height);
            SKBitmap paddingSrc = OcrUtils.MakePadding(originSrc, padding);
            using (var fs = new FileStream("padding.bmp", FileMode.Create))
            {
                paddingSrc.Encode(fs, SKEncodedImageFormat.Png, 100);
            }

            ScaleParam scale = ScaleParam.GetScaleParam(paddingSrc, resize);

            return DetectOnce(paddingSrc, paddingRect, scale, boxScoreThresh, boxThresh, unClipRatio, doAngle, mostAngle);


            /*

            Mat originSrc = CvInvoke.Imread(img, ImreadModes.Color);//default : BGR
            int originMaxSide = Math.Max(originSrc.Cols, originSrc.Rows);

            int resize;
            if (maxSideLen <= 0 || maxSideLen > originMaxSide)
            {
                resize = originMaxSide;
            }
            else
            {
                resize = maxSideLen;
            }
            resize += 2 * padding;
            Rectangle paddingRect = new Rectangle(padding, padding, originSrc.Cols, originSrc.Rows);
            Mat paddingSrc = OcrUtils.MakePadding(originSrc, padding);

            ScaleParam scale = ScaleParam.GetScaleParam(paddingSrc, resize);

            return DetectOnce(paddingSrc, paddingRect, scale, boxScoreThresh, boxThresh, unClipRatio, doAngle, mostAngle);
            */
        }

        private OcrResult DetectOnce(Mat src, Rectangle originRect, ScaleParam scale, float boxScoreThresh, float boxThresh,
            float unClipRatio, bool doAngle, bool mostAngle)
        {
            Mat textBoxPaddingImg = src.Clone();
            int thickness = OcrUtils.GetThickness(src);
            Console.WriteLine("=====Start detect=====");
            var startTicks = DateTime.Now.Ticks;

            Console.WriteLine("---------- step: dbNet getTextBoxes ----------");
            var textBoxes = dbNet.GetTextBoxes(src, scale, boxScoreThresh, boxThresh, unClipRatio);
            var dbNetTime = (DateTime.Now.Ticks - startTicks) / 10000F;

            Console.WriteLine($"TextBoxesSize({textBoxes.Count})");
            textBoxes.ForEach(x => Console.WriteLine(x));
            //Console.WriteLine($"dbNetTime({dbNetTime}ms)");

            Console.WriteLine("---------- step: drawTextBoxes ----------");
            OcrUtils.DrawTextBoxes(textBoxPaddingImg, textBoxes, thickness);
            //CvInvoke.Imshow("ResultPadding", textBoxPaddingImg);

            //---------- getPartImages ----------
            List<Mat> partImages = OcrUtils.GetPartImages(src, textBoxes);
            if (isPartImg)
            {
                for (int i = 0; i < partImages.Count; i++)
                {
                    CvInvoke.Imshow($"PartImg({i})", partImages[i]);
                }
            }

            Console.WriteLine("---------- step: angleNet getAngles ----------");
            List<Angle> angles = angleNet.GetAngles(partImages, doAngle, mostAngle);
            //angles.ForEach(x => Console.WriteLine(x));

            //Rotate partImgs
            for (int i = 0; i < partImages.Count; ++i)
            {
                if (angles[i].Index == 1)
                {
                    partImages[i] = OcrUtils.MatRotateClockWise180(partImages[i]);
                }
                if (isDebugImg)
                {
                    CvInvoke.Imshow($"DebugImg({i})", partImages[i]);
                }
            }

            Console.WriteLine("---------- step: crnnNet getTextLines ----------");
            List<TextLine> textLines = crnnNet.GetTextLines(partImages);
            //textLines.ForEach(x => Console.WriteLine(x));

            List<TextBlock> textBlocks = new List<TextBlock>();
            for (int i = 0; i < textLines.Count; ++i)
            {
                TextBlock textBlock = new TextBlock();
                textBlock.BoxPoints = textBoxes[i].Points;
                textBlock.BoxScore = textBoxes[i].Score;
                textBlock.AngleIndex = angles[i].Index;
                textBlock.AngleScore = angles[i].Score;
                textBlock.AngleTime = angles[i].Time;
                textBlock.Text = textLines[i].Text;
                textBlock.CharScores = textLines[i].CharScores;
                textBlock.CrnnTime = textLines[i].Time;
                textBlock.BlockTime = angles[i].Time + textLines[i].Time;
                textBlocks.Add(textBlock);
            }
            //textBlocks.ForEach(x => Console.WriteLine(x));

            var endTicks = DateTime.Now.Ticks;
            var fullDetectTime = (endTicks - startTicks) / 10000F;
            //Console.WriteLine($"fullDetectTime({fullDetectTime}ms)");

            //cropped to original size
            Mat boxImg = new Mat(textBoxPaddingImg, originRect);

            StringBuilder strRes = new StringBuilder();
            textBlocks.ForEach(x => strRes.AppendLine(x.Text));

            OcrResult ocrResult = new OcrResult();
            ocrResult.TextBlocks = textBlocks;
            ocrResult.DbNetTime = dbNetTime;
            ocrResult.BoxImg = boxImg;
            ocrResult.DetectTime = fullDetectTime;
            ocrResult.StrRes = strRes.ToString();

            return ocrResult;
        }

        private OcrResult DetectOnce(SKBitmap src, Rectangle originRect, ScaleParam scale, float boxScoreThresh, float boxThresh,
            float unClipRatio, bool doAngle, bool mostAngle)
        {
            SKBitmap textBoxPaddingImg = src; //.Clone();
            int thickness = OcrUtils.GetThickness(src);
            Console.WriteLine("=====Start detect=====");
            var startTicks = DateTime.Now.Ticks;

            Console.WriteLine("---------- step: dbNet getTextBoxes ----------");
            var textBoxes = dbNet.GetTextBoxes(src, scale, boxScoreThresh, boxThresh, unClipRatio);
            var dbNetTime = (DateTime.Now.Ticks - startTicks) / 10000F;

            Console.WriteLine($"TextBoxesSize({textBoxes.Count})");
            textBoxes.ForEach(x => Console.WriteLine(x));

            //---------- getPartImages ----------
            SKBitmap[] partImages = OcrUtils.GetPartImages(src, textBoxes).ToArray();

            Console.WriteLine("---------- step: angleNet getAngles ----------");
            List<Angle> angles = angleNet.GetAngles(partImages, doAngle, mostAngle);
            //angles.ForEach(x => Console.WriteLine(x));

            //Rotate partImgs
            for (int i = 0; i < partImages.Length; ++i)
            {
                if (angles[i].Index == 1)
                {
                    partImages[i] = OcrUtils.MatRotateClockWise180(partImages[i]);
                }
            }

            Console.WriteLine("---------- step: crnnNet getTextLines ----------");
            List<TextLine> textLines = crnnNet.GetTextLines(partImages);

            List<TextBlock> textBlocks = new List<TextBlock>();
            for (int i = 0; i < textLines.Count; ++i)
            {
                TextBlock textBlock = new TextBlock();
                textBlock.BoxPoints = textBoxes[i].Points;
                textBlock.BoxScore = textBoxes[i].Score;
                textBlock.AngleIndex = angles[i].Index;
                textBlock.AngleScore = angles[i].Score;
                textBlock.AngleTime = angles[i].Time;
                textBlock.Text = textLines[i].Text;
                textBlock.CharScores = textLines[i].CharScores;
                textBlock.CrnnTime = textLines[i].Time;
                textBlock.BlockTime = angles[i].Time + textLines[i].Time;
                textBlocks.Add(textBlock);
            }
            //textBlocks.ForEach(x => Console.WriteLine(x));

            var endTicks = DateTime.Now.Ticks;
            var fullDetectTime = (endTicks - startTicks) / 10000F;
            //Console.WriteLine($"fullDetectTime({fullDetectTime}ms)");

            //cropped to original size
            //Mat boxImg = new Mat(textBoxPaddingImg, originRect);

            StringBuilder strRes = new StringBuilder();
            textBlocks.ForEach(x => strRes.AppendLine(x.Text));

            OcrResult ocrResult = new OcrResult();
            ocrResult.TextBlocks = textBlocks;
            ocrResult.DbNetTime = dbNetTime;
            //ocrResult.BoxImg = boxImg;
            ocrResult.DetectTime = fullDetectTime;
            ocrResult.StrRes = strRes.ToString();

            return ocrResult;
        }
    }
}
