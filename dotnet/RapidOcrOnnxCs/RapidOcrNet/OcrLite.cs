using System.Diagnostics;
using System.Text;
using SkiaSharp;

namespace RapidOcrNet
{
    public sealed class OcrLite
    {
        private readonly DbNet _dbNet = new DbNet();
        private readonly AngleNet _angleNet = new AngleNet();
        private readonly CrnnNet _crnnNet = new CrnnNet();
        
        public void InitModels(string detPath, string clsPath, string recPath, string keysPath, int numThread)
        {
            try
            {
                _dbNet.InitModel(detPath, numThread);
                _angleNet.InitModel(clsPath, numThread);
                _crnnNet.InitModel(recPath, keysPath, numThread);
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine(ex.Message + ex.StackTrace);
                throw;
            }
        }

        public OcrResult Detect(string img, int padding, int maxSideLen, float boxScoreThresh, float boxThresh,
            float unClipRatio, bool doAngle, bool mostAngle)
        {
            using (SKBitmap originSrc = SKBitmap.Decode(img))
            {
                return Detect(originSrc, padding, maxSideLen, boxScoreThresh, boxThresh, unClipRatio, doAngle,
                    mostAngle);
            }
        }

        public OcrResult Detect(SKBitmap originSrc, int padding, int maxSideLen, float boxScoreThresh, float boxThresh,
            float unClipRatio, bool doAngle, bool mostAngle)
        {
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
            var paddingRect = new SKRectI(padding, padding, originSrc.Width + padding, originSrc.Height + padding);
            using (SKBitmap paddingSrc = OcrUtils.MakePadding(originSrc, padding))
            {
                return DetectOnce(paddingSrc, paddingRect, ScaleParam.GetScaleParam(paddingSrc, resize),
                    boxScoreThresh, boxThresh, unClipRatio, doAngle, mostAngle);
            }
        }

        private OcrResult DetectOnce(SKBitmap src, SKRectI originRect, ScaleParam scale, float boxScoreThresh,
            float boxThresh, float unClipRatio, bool doAngle, bool mostAngle)
        {
            System.Diagnostics.Debug.WriteLine("=====Start detect=====");
            var sw = Stopwatch.StartNew();

            System.Diagnostics.Debug.WriteLine("---------- step: dbNet getTextBoxes ----------");
            var textBoxes = _dbNet.GetTextBoxes(src, scale, boxScoreThresh, boxThresh, unClipRatio);
            var dbNetTime = sw.ElapsedMilliseconds;

            System.Diagnostics.Debug.WriteLine($"TextBoxesSize({textBoxes.Count})");
#if DEBUG
            foreach (var x in  textBoxes)
            {
                Debug.WriteLine(x);
            }
#endif

            //---------- getPartImages ----------
            SKBitmap[] partImages = OcrUtils.GetPartImages(src, textBoxes).ToArray();

            System.Diagnostics.Debug.WriteLine("---------- step: angleNet getAngles ----------");
            List<Angle> angles = _angleNet.GetAngles(partImages, doAngle, mostAngle);

            //Rotate partImgs
            for (int i = 0; i < partImages.Length; ++i)
            {
                if (angles[i].Index == 1)
                {
                    partImages[i] = OcrUtils.MatRotateClockWise180(partImages[i]);
                }
            }

            System.Diagnostics.Debug.WriteLine("---------- step: crnnNet getTextLines ----------");
            List<TextLine> textLines = _crnnNet.GetTextLines(partImages);

            foreach (var bmp in partImages)
            {
                bmp.Dispose();
            }

            var textBlocks = new TextBlock[textLines.Count];
            for (int i = 0; i < textLines.Count; ++i)
            {
                var textBox = textBoxes[i];
                var angle = angles[i];
                var textLine = textLines[i];

                textBlocks[i] = new TextBlock
                {
                    BoxPoints = textBox.Points,
                    BoxScore = textBox.Score,
                    AngleIndex = angle.Index,
                    AngleScore = angle.Score,
                    AngleTime = angle.Time,
                    Text = textLine.Text,
                    CharScores = textLine.CharScores,
                    CrnnTime = textLine.Time,
                    BlockTime = angle.Time + textLine.Time
                };
            }

            var fullDetectTime = sw.ElapsedMilliseconds;

            var strRes = new StringBuilder();
            foreach (var x in textBlocks)
            {
                strRes.AppendLine(x.Text);
            }

            return new OcrResult
            {
                TextBlocks = textBlocks,
                DbNetTime = dbNetTime,
                DetectTime = fullDetectTime,
                StrRes = strRes.ToString()
            };
        }
    }
}
