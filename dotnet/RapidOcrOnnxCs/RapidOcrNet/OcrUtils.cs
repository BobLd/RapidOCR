using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;

namespace RapidOcrNet
{
    internal static class OcrUtils
    {
        public static Tensor<float> SubtractMeanNormalize(SKBitmap src, float[] meanVals, float[] normVals)
        {
            int cols = src.Width;
            int rows = src.Height;
            int channels = src.BytesPerPixel;

            const int expChannels = 3; // Size of meanVals

            Tensor<float> inputTensor = new DenseTensor<float>([1, expChannels, rows, cols]);

            ReadOnlySpan<byte> span = src.GetPixelSpan();
            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < cols; c++)
                {
                    int i = r * cols + c;
                    for (int ch = 0; ch < expChannels; ch++)
                    {
                        byte value = span[i * channels + ch];
                        inputTensor[0, ch, r, c] = value * normVals[ch] - meanVals[ch] * normVals[ch];
                    }
                }
            }

            return inputTensor;
        }

        public static SKBitmap MakePadding(SKBitmap src, int padding)
        {
            if (padding <= 0)
            {
                return src;
            }

            SKImageInfo info = src.Info;

            info.Width += 2 * padding;
            info.Height += 2 * padding;

            SKBitmap newBmp = new SKBitmap(info);
            using (SKCanvas canvas = new SKCanvas(newBmp))
            {
                canvas.Clear(SKColors.White);
                canvas.DrawBitmap(src, new SKPoint(padding, padding));
            }

            return newBmp;
        }

        public static int GetThickness(SKBitmap boxImg)
        {
            int minSize = boxImg.Width > boxImg.Height ? boxImg.Height : boxImg.Width;
            return minSize / 1000 + 2;
        }

        public static IEnumerable<SKBitmap> GetPartImages(SKBitmap src, List<TextBox> textBoxes)
        {
            for (int i = 0; i < textBoxes.Count; ++i)
            {
                yield return GetRotateCropImage(src, textBoxes[i].Points);
            }
        }

        public static SKMatrix GetPerspectiveTransform(SKPoint topLeft, SKPoint topRight, SKPoint botRight, SKPoint botLeft,
            float width, float height)
        {
            // https://stackoverflow.com/questions/48416118/perspective-transform-in-skia

            float x1 = topLeft.X;
            float y1 = topLeft.Y;
            float x2 = topRight.X;
            float y2 = topRight.Y;
            float x3 = botRight.X;
            float y3 = botRight.Y;

            float x4 = botLeft.X;
            float y4 = botLeft.Y;

            float w = width;
            float h = height;

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

        public static SKBitmap GetRotateCropImage(SKBitmap src, List<SKPointI> box)
        {
            Span<SKPointI> points = box.ToArray();

            ReadOnlySpan<int> collectX = stackalloc int[] { box[0].X, box[1].X, box[2].X, box[3].X };
            int left = int.MaxValue;
            int right = int.MinValue;
            foreach (var v in collectX)
            {
                if (v < left)
                {
                    left = v;
                }
                else if (v > right)
                {
                    right = v;
                }
            }

            ReadOnlySpan<int> collectY = stackalloc int[] { box[0].Y, box[1].Y, box[2].Y, box[3].Y };
            int top = int.MaxValue;
            int bottom = int.MinValue;
            foreach (var v in collectY)
            {
                if (v < top)
                {
                    left = v;
                }
                else if (v > bottom)
                {
                    right = v;
                }
            }

            SKRectI rect = new SKRectI(left, top, right, bottom);

            var info = src.Info;
            info.Width = rect.Width;
            info.Height = rect.Height;

            SKBitmap imgCrop = new SKBitmap(info);
            if (!src.ExtractSubset(imgCrop, rect))
            {
                throw new Exception("Could not extract subset.");
            }

            for (int i = 0; i < points.Length; i++)
            {
                var pt = points[i];
                pt.X -= left;
                pt.Y -= top;
                points[i] = pt;
            }

            int imgCropWidth = (int)Math.Sqrt(Math.Pow(points[0].X - points[1].X, 2) +
                                              Math.Pow(points[0].Y - points[1].Y, 2));
            int imgCropHeight = (int)Math.Sqrt(Math.Pow(points[0].X - points[3].X, 2) +
                                               Math.Pow(points[0].Y - points[3].Y, 2));

            var ptsSrc0Sk = new SKPoint(points[0].X, points[0].Y);
            var ptsSrc1Sk = new SKPoint(points[1].X, points[1].Y);
            var ptsSrc2Sk = new SKPoint(points[2].X, points[2].Y);
            var ptsSrc3Sk = new SKPoint(points[3].X, points[3].Y);

            var m = GetPerspectiveTransform(ptsSrc0Sk, ptsSrc1Sk, ptsSrc2Sk, ptsSrc3Sk, imgCropWidth, imgCropHeight);

            if (m.IsIdentity)
            {
                if (imgCrop.Height >= imgCrop.Width * 1.5)
                {
                    return MatRotateClockWise90(imgCrop);
                }

                return imgCrop;
            }

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

            if (partImg.Height >= partImg.Width * 1.5)
            {
                return MatRotateClockWise90(partImg);
            }

            return partImg;
        }

        public static SKBitmap MatRotateClockWise180(SKBitmap src)
        {
            var rotated = new SKBitmap(src.Info);

            using (var canvas = new SKCanvas(rotated))
            {
                canvas.Translate(rotated.Width, rotated.Height);
                canvas.RotateDegrees(180);
                canvas.DrawBitmap(src, 0, 0);
            }

            return rotated;
        }

        public static SKBitmap MatRotateClockWise90(SKBitmap src)
        {
            var rotated = new SKBitmap(src.Info);

            using (var canvas = new SKCanvas(rotated))
            {
                canvas.Translate(rotated.Width, 0);
                canvas.RotateDegrees(90);
                canvas.DrawBitmap(src, 0, 0);
            }

            return rotated;
        }
    }
}
