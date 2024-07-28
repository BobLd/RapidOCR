using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using SkiaSharp;

namespace OcrLiteLib
{
    class OcrUtils
    {
        public static Tensor<float> SubstractMeanNormalize(Mat src, float[] meanVals, float[] normVals)
        {
            int cols = src.Cols;
            int rows = src.Rows;
            int channels = src.NumberOfChannels;
            Image<Rgb, byte> srcImg = src.ToImage<Rgb, byte>();
            byte[,,] imgData = srcImg.Data;
            Tensor<float> inputTensor = new DenseTensor<float>(new[] { 1, channels, rows, cols });
            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < cols; c++)
                {
                    for (int ch = 0; ch < channels; ch++)
                    {
                        var value = imgData[r, c, ch];
                        float data = (float)(value * normVals[ch] - meanVals[ch] * normVals[ch]);
                        inputTensor[0, ch, r, c] = data;
                    }
                }
            }
            return inputTensor;
        }

        public static Tensor<float> SubstractMeanNormalize(SKBitmap src, float[] meanVals, float[] normVals)
        {
            int cols = src.Width;
            int rows = src.Height;
            int channels = src.BytesPerPixel; //.NumberOfChannels;

            int expChannels = 3;

            ReadOnlySpan<byte> span = src.GetPixelSpan();

            Tensor<float> inputTensor = new DenseTensor<float>(new[] { 1, expChannels, rows, cols });

            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < cols; c++)
                {
                    int i = r * cols + c;
                    for (int ch = 0; ch < expChannels; ch++)
                    {
                        byte value = span[i * (channels) + ch];
                        inputTensor[0, ch, r, c] = (float)(value * normVals[ch] - meanVals[ch] * normVals[ch]);
                    }
                }
            }

            return inputTensor;
        }

        public static Mat MakePadding(Mat src, int padding)
        {
            if (padding <= 0) return src;
            MCvScalar paddingScalar = new MCvScalar(255, 255, 255);
            Mat paddingSrc = new Mat();
            CvInvoke.CopyMakeBorder(src, paddingSrc, padding, padding, padding, padding, BorderType.Isolated, paddingScalar);
            return paddingSrc;
        }

        public static SKBitmap MakePadding(SKBitmap src, int padding)
        {
            if (padding <= 0)
            {
                return src;
            }

            SKImageInfo info = src.Info;

            info.Width +=  2 * padding;
            info.Height += 2 * padding;

            SKBitmap newBmp = new SKBitmap(info);
            using (SKCanvas canvas = new SKCanvas(newBmp))
            {
                canvas.Clear(SKColors.White);
                canvas.DrawBitmap(src, new SKPoint(padding, padding));
            }

            return newBmp;
        }

        public static int GetThickness(Mat boxImg)
        {
            int minSize = boxImg.Cols > boxImg.Rows ? boxImg.Rows : boxImg.Cols;
            int thickness = minSize / 1000 + 2;
            return thickness;
        }

        public static int GetThickness(SKBitmap boxImg)
        {
            int minSize = boxImg.Width > boxImg.Height ? boxImg.Height : boxImg.Width;
            int thickness = minSize / 1000 + 2;
            return thickness;
        }

        public static void DrawTextBox(Mat boxImg, List<Point> box, int thickness)
        {
            if (box == null || box.Count == 0)
            {
                return;
            }
            var color = new MCvScalar(0, 0, 255);//B(0) G(0) R(255)
            CvInvoke.Line(boxImg, box[0], box[1], color, thickness);
            CvInvoke.Line(boxImg, box[1], box[2], color, thickness);
            CvInvoke.Line(boxImg, box[2], box[3], color, thickness);
            CvInvoke.Line(boxImg, box[3], box[0], color, thickness);
        }

        public static void DrawTextBoxes(Mat src, List<TextBox> textBoxes, int thickness)
        {
            for (int i = 0; i < textBoxes.Count; i++)
            {
                TextBox t = textBoxes[i];
                DrawTextBox(src, t.Points, thickness);
            }
        }

        public static List<SKBitmap> GetPartImages(SKBitmap src, List<TextBox> textBoxes)
        {
            List<SKBitmap> partImages = new List<SKBitmap>();
            for (int i = 0; i < textBoxes.Count; ++i)
            {
                SKBitmap partImg = GetRotateCropImage(src, textBoxes[i].Points);
                //Mat partImg = new Mat();
                //GetRoiFromBox(src, partImg, textBoxes[i].Points);
                partImages.Add(partImg);
            }
            return partImages;
        }


        public static List<Mat> GetPartImages(Mat src, List<TextBox> textBoxes)
        {
            List<Mat> partImages = new List<Mat>();
            for (int i = 0; i < textBoxes.Count; ++i)
            {
                Mat partImg = GetRotateCropImage(src, textBoxes[i].Points);
                //Mat partImg = new Mat();
                //GetRoiFromBox(src, partImg, textBoxes[i].Points);
                partImages.Add(partImg);
            }
            return partImages;
        }

        public static SKMatrix GetPerspectiveTransform(SKPoint topLeft, SKPoint topRight, SKPoint botRight, SKPoint botLeft, float width, float height)
        {
            // https://stackoverflow.com/questions/48416118/perspective-transform-in-skia

            (float x1, float y1) = (topLeft.X, topLeft.Y);
            (float x2, float y2) = (topRight.X, topRight.Y);
            (float x3, float y3) = (botRight.X, botRight.Y);
            (float x4, float y4) = (botLeft.X, botLeft.Y);
            (float w, float h) = (width, height);

            float scaleX = (y1 * x2 * x4 - x1 * y2 * x4 + x1 * y3 * x4 - x2 * y3 * x4 - y1 * x2 * x3 + x1 * y2 * x3 - x1 * y4 * x3 + x2 * y4 * x3) / (x2 * y3 * w + y2 * x4 * w - y3 * x4 * w - x2 * y4 * w - y2 * w * x3 + y4 * w * x3);
            float skewX = (-x1 * x2 * y3 - y1 * x2 * x4 + x2 * y3 * x4 + x1 * x2 * y4 + x1 * y2 * x3 + y1 * x4 * x3 - y2 * x4 * x3 - x1 * y4 * x3) / (x2 * y3 * h + y2 * x4 * h - y3 * x4 * h - x2 * y4 * h - y2 * h * x3 + y4 * h * x3);
            float transX = x1;
            float skewY = (-y1 * x2 * y3 + x1 * y2 * y3 + y1 * y3 * x4 - y2 * y3 * x4 + y1 * x2 * y4 - x1 * y2 * y4 - y1 * y4 * x3 + y2 * y4 * x3) / (x2 * y3 * w + y2 * x4 * w - y3 * x4 * w - x2 * y4 * w - y2 * w * x3 + y4 * w * x3);
            float scaleY = (-y1 * x2 * y3 - y1 * y2 * x4 + y1 * y3 * x4 + x1 * y2 * y4 - x1 * y3 * y4 + x2 * y3 * y4 + y1 * y2 * x3 - y2 * y4 * x3) / (x2 * y3 * h + y2 * x4 * h - y3 * x4 * h - x2 * y4 * h - y2 * h * x3 + y4 * h * x3);
            float transY = y1;
            float persp0 = (x1 * y3 - x2 * y3 + y1 * x4 - y2 * x4 - x1 * y4 + x2 * y4 - y1 * x3 + y2 * x3) / (x2 * y3 * w + y2 * x4 * w - y3 * x4 * w - x2 * y4 * w - y2 * w * x3 + y4 * w * x3);
            float persp1 = (-y1 * x2 + x1 * y2 - x1 * y3 - y2 * x4 + y3 * x4 + x2 * y4 + y1 * x3 - y4 * x3) / (x2 * y3 * h + y2 * x4 * h - y3 * x4 * h - x2 * y4 * h - y2 * h * x3 + y4 * h * x3);
            float persp2 = 1;

            return new SKMatrix(scaleX, skewX, transX, skewY, scaleY, transY, persp0, persp1, persp2);
        }

        public static SKBitmap GetRotateCropImage(SKBitmap src, List<Point> box)
        {
            SKBitmap image = new SKBitmap(src.Info);
            src.CopyTo(image);
            List<Point> points = new List<Point>();
            points.AddRange(box);

            int[] collectX = { box[0].X, box[1].X, box[2].X, box[3].X };
            int[] collectY = { box[0].Y, box[1].Y, box[2].Y, box[3].Y };
            int left = collectX.Min();
            int right = collectX.Max();
            int top = collectY.Min();
            int bottom = collectY.Max();

            SKRectI rect = new SKRectI(left, top, right, bottom);

            var info = src.Info;
            info.Width = rect.Width;
            info.Height = rect.Height;

            SKBitmap imgCrop = new SKBitmap(info);

            bool success = src.ExtractSubset(imgCrop, rect);

            for (int i = 0; i < points.Count; i++)
            {
                var pt = points[i];
                pt.X -= left;
                pt.Y -= top;
                points[i] = pt;
            }

            int imgCropWidth = (int)(Math.Sqrt(Math.Pow(points[0].X - points[1].X, 2) +
                                        Math.Pow(points[0].Y - points[1].Y, 2)));
            int imgCropHeight = (int)(Math.Sqrt(Math.Pow(points[0].X - points[3].X, 2) +
                                         Math.Pow(points[0].Y - points[3].Y, 2)));

            var ptsDst0 = new PointF(0, 0);
            var ptsDst1 = new PointF(imgCropWidth, 0);
            var ptsDst2 = new PointF(imgCropWidth, imgCropHeight);
            var ptsDst3 = new PointF(0, imgCropHeight);

            PointF[] ptsDst = { ptsDst0, ptsDst1, ptsDst2, ptsDst3 };

            var ptsSrc0 = new PointF(points[0].X, points[0].Y);
            var ptsSrc1 = new PointF(points[1].X, points[1].Y);
            var ptsSrc2 = new PointF(points[2].X, points[2].Y);
            var ptsSrc3 = new PointF(points[3].X, points[3].Y);

            PointF[] ptsSrc = { ptsSrc0, ptsSrc1, ptsSrc2, ptsSrc3 };

            var ptsSrc0Sk = new SKPoint(points[0].X, points[0].Y);
            var ptsSrc1Sk = new SKPoint(points[1].X, points[1].Y);
            var ptsSrc2Sk = new SKPoint(points[2].X, points[2].Y);
            var ptsSrc3Sk = new SKPoint(points[3].X, points[3].Y);

            var m = GetPerspectiveTransform(ptsSrc0Sk, ptsSrc1Sk, ptsSrc2Sk, ptsSrc3Sk, imgCropWidth, imgCropHeight);

            var info2 = imgCrop.Info;
            info2.Width = imgCropWidth;
            info2.Height = imgCropHeight;

            var partImg = new SKBitmap(info2);
            using (var canvas = new SKCanvas(partImg))
            {
                canvas.SetMatrix(m);
                canvas.DrawBitmap(imgCrop, 0, 0);
                canvas.Restore();
            }

            /*
            Mat M = CvInvoke.GetPerspectiveTransform(ptsSrc, ptsDst);

            Mat partImg = new Mat();
            CvInvoke.WarpPerspective(imgCrop, partImg, M,
                                new Size(imgCropWidth, imgCropHeight), Inter.Nearest, Warp.Default,
                               BorderType.Replicate);
            */

            if (partImg.Height >= partImg.Width * 1.5)
            {
                throw new NotImplementedException("TODO");
                /*
                Mat srcCopy = new Mat();
                CvInvoke.Transpose(partImg, srcCopy);
                CvInvoke.Flip(srcCopy, srcCopy, 0);
                return srcCopy;
                */
            }
            else
            {
                return partImg;
            }
            
        }

        public static Mat GetRotateCropImage(Mat src, List<Point> box)
        {
            Mat image = new Mat();
            src.CopyTo(image);
            List<Point> points = new List<Point>();
            points.AddRange(box);

            int[] collectX = { box[0].X, box[1].X, box[2].X, box[3].X };
            int[] collectY = { box[0].Y, box[1].Y, box[2].Y, box[3].Y };
            int left = collectX.Min();
            int right = collectX.Max();
            int top = collectY.Min();
            int bottom = collectY.Max();

            Rectangle rect = new Rectangle(left, top, right - left, bottom - top);
            Mat imgCrop = new Mat(image, rect);

            for (int i = 0; i < points.Count; i++)
            {
                var pt = points[i];
                pt.X -= left;
                pt.Y -= top;
                points[i] = pt;
            }

            int imgCropWidth = (int)(Math.Sqrt(Math.Pow(points[0].X - points[1].X, 2) +
                                        Math.Pow(points[0].Y - points[1].Y, 2)));
            int imgCropHeight = (int)(Math.Sqrt(Math.Pow(points[0].X - points[3].X, 2) +
                                         Math.Pow(points[0].Y - points[3].Y, 2)));

            var ptsDst0 = new PointF(0, 0);
            var ptsDst1 = new PointF(imgCropWidth, 0);
            var ptsDst2 = new PointF(imgCropWidth, imgCropHeight);
            var ptsDst3 = new PointF(0, imgCropHeight);

            PointF[] ptsDst = { ptsDst0, ptsDst1, ptsDst2, ptsDst3 };


            var ptsSrc0 = new PointF(points[0].X, points[0].Y);
            var ptsSrc1 = new PointF(points[1].X, points[1].Y);
            var ptsSrc2 = new PointF(points[2].X, points[2].Y);
            var ptsSrc3 = new PointF(points[3].X, points[3].Y);

            PointF[] ptsSrc = { ptsSrc0, ptsSrc1, ptsSrc2, ptsSrc3 };

            Mat M = CvInvoke.GetPerspectiveTransform(ptsSrc, ptsDst);

            Mat partImg = new Mat();
            CvInvoke.WarpPerspective(imgCrop, partImg, M,
                                new Size(imgCropWidth, imgCropHeight), Inter.Nearest, Warp.Default,
                               BorderType.Replicate);

            if (partImg.Rows >= partImg.Cols * 1.5)
            {
                Mat srcCopy = new Mat();
                CvInvoke.Transpose(partImg, srcCopy);
                CvInvoke.Flip(srcCopy, srcCopy, 0);
                return srcCopy;
            }
            else
            {
                return partImg;
            }
        }

        public static Mat MatRotateClockWise180(Mat src)
        {
            CvInvoke.Flip(src, src, FlipType.Vertical);
            CvInvoke.Flip(src, src, FlipType.Horizontal);
            return src;
        }

        public static Mat MatRotateClockWise90(Mat src)
        {
            CvInvoke.Rotate(src, src, RotateFlags.Rotate90CounterClockwise);
            return src;
        }

    }
}

