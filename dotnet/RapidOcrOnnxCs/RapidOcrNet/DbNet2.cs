using System.Runtime.InteropServices;
using Clipper2Lib;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OcrLib;
using PContourNet;
using SkiaSharp;

namespace OcrLiteLib
{
    internal sealed class DbNet2
    {
        private readonly float[] MeanValues = [0.485F * 255F, 0.456F * 255F, 0.406F * 255F];
        private readonly float[] NormValues = [1.0F / 0.229F / 255.0F, 1.0F / 0.224F / 255.0F, 1.0F / 0.225F / 255.0F];

        private InferenceSession dbNet;

        private List<string> inputNames;

        public DbNet2() { }

        ~DbNet2()
        {
            dbNet.Dispose();
        }

        public void InitModel(string path, int numThread)
        {
            try
            {
                var op = new SessionOptions
                {
                    GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
                    InterOpNumThreads = numThread,
                    IntraOpNumThreads = numThread
                };

                dbNet = new InferenceSession(path, op);
                inputNames = dbNet.InputMetadata.Keys.ToList();
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine(ex.Message + ex.StackTrace);
                throw;
            }
        }

        public List<TextBox> GetTextBoxes(SKBitmap src, ScaleParam scale, float boxScoreThresh, float boxThresh,
            float unClipRatio)
        {
            Tensor<float> inputTensors;
            using (var srcResize = src.Resize(new SKSizeI(scale.DstWidth, scale.DstHeight), SKFilterQuality.High))
            {
                inputTensors = OcrUtils.SubtractMeanNormalize(srcResize, MeanValues, NormValues);
            }

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(inputNames[0], inputTensors)
            };

            try
            {
                using (IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = dbNet.Run(inputs))
                {
                    var resultsArray = results.ToArray();
                    System.Diagnostics.Debug.WriteLine(resultsArray);
                    return GetTextBoxes(resultsArray, scale.DstHeight, scale.DstWidth, scale, boxScoreThresh,
                        boxThresh, unClipRatio);
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine(ex.Message + ex.StackTrace);
            }

            return null;
        }

        private static SKPointI[][] FindContours(ReadOnlySpan<byte> array, int rows, int cols)
        {
            var v = array.ToArray().Select(b => (int)(b / byte.MaxValue)).ToArray();
            var contours = PContour.FindContours(v.AsSpan(), cols, rows);
            return contours.Select(c => PContour.ApproxPolyDP(c.points.ToArray(), 1).ToArray()).ToArray();
        }

        private static bool TryFindIndex(Dictionary<int, int> link, int offset, out int index)
        {
            bool found = false;
            index = offset;
            while (link.TryGetValue(index, out int newIndex))
            {
                found = true;
                if (index == newIndex) break;
                index = newIndex;
            }
            return found;
        }

        private static List<TextBox> GetTextBoxes(DisposableNamedOnnxValue[] outputTensor, int rows, int cols, ScaleParam s, float boxScoreThresh, float boxThresh, float unClipRatio)
        {
            const float maxSideThresh = 3.0f; // Long Edge Threshold
            List<TextBox> rsBoxes = new List<TextBox>();

            //-----Data preparation-----
            float[] predData = outputTensor[0].AsEnumerable<float>().ToArray();

            var gray8 = new SKImageInfo()
            {
                Height = rows,
                Width = cols,
                AlphaType = SKAlphaType.Opaque,
                ColorType = SKColorType.Gray8
            };

            var crop = new SKRectI(0, 0, cols, rows);

            Span<byte> rawPixels = new byte[predData.Length];
            for (int i = 0; i < predData.Length; i++)
            {
                // Thresholding
                rawPixels[i] = predData[i] > boxThresh ? byte.MaxValue : byte.MinValue;
            }

            const int dilateRadius = 2;

            SKPointI[][] contours;
            using (var skImage = SKImage.FromPixelCopy(gray8, rawPixels))
            using (var filter = SKImageFilter.CreateDilate(dilateRadius, dilateRadius))
            using (var dilated = skImage.ApplyImageFilter(filter, crop, crop, out SKRectI _, out SKPointI _)) // Dilate
            using (var croppedDilatedSubset = dilated.Subset(crop)) // Trim image due to dilate
            {
                IntPtr buffer = Marshal.AllocHGlobal(gray8.BytesSize);
                try
                {
                    croppedDilatedSubset.ReadPixels(gray8, buffer);
                    byte[] bytes = new byte[rawPixels.Length];

                    Marshal.Copy(buffer, bytes, 0, rawPixels.Length);

                    contours = FindContours(bytes, rows, cols);
                }
                finally
                {
                    Marshal.FreeHGlobal(buffer);
                }
            }

            using (SKImage predMatImage = SKImage.FromPixelCopy(gray8, predData.Select(b => Convert.ToByte(b * 255)).ToArray()))
            {
                for (int i = 0; i < contours.Length; i++)
                {
                    if (contours[i].Length <= 2)
                    {
                        continue;
                    }

                    float maxSide = 0;
                    List<SKPoint> minBox = GetMiniBox(contours[i], out maxSide);
                    if (maxSide < maxSideThresh)
                    {
                        continue;
                    }

                    double score = GetScore(contours[i], predMatImage);
                    if (score < boxScoreThresh)
                    {
                        continue;
                    }

                    List<SKPointI> clipBox = Unclip(minBox, unClipRatio);
                    if (clipBox == null)
                    {
                        continue;
                    }

                    List<SKPoint> clipMinBox = GetMiniBox(clipBox, out maxSide);
                    if (maxSide < maxSideThresh + 2)
                    {
                        continue;
                    }

                    List<SKPointI> finalPoints = new List<SKPointI>();
                    foreach (var item in clipMinBox)
                    {
                        int x = (int)(item.X / s.ScaleWidth);
                        int ptx = Math.Min(Math.Max(x, 0), s.SrcWidth);

                        int y = (int)(item.Y / s.ScaleHeight);
                        int pty = Math.Min(Math.Max(y, 0), s.SrcHeight);

                        SKPointI dstPt = new SKPointI(ptx, pty);
                        finalPoints.Add(dstPt);
                    }

                    var textBox = new TextBox
                    {
                        Score = (float)score,
                        Points = finalPoints
                    };
                    rsBoxes.Add(textBox);
                }
            }

            rsBoxes.Reverse();
            return rsBoxes;
        }

        private static List<SKPoint> GetMiniBox(List<SKPointI> contours, out float minEdgeSize)
        {
            return GetMiniBox(contours.ToArray(), out minEdgeSize);
        }

        private static List<SKPoint> GetMiniBox(SKPointI[] contours, out float minEdgeSize)
        {
            List<SKPoint> box = new List<SKPoint>();

            SKPoint[] points = GeometryExtensions.MinimumAreaRectangle(contours);

            var size = GeometryExtensions.GetSize(points);
            minEdgeSize = Math.Min(size.width, size.height);

            List<SKPoint> thePoints = new List<SKPoint>(points);
            thePoints.Sort(CompareByX);

            int index_1 = 0, index_2 = 1, index_3 = 2, index_4 = 3;
            if (thePoints[1].Y > thePoints[0].Y)
            {
                index_1 = 0;
                index_4 = 1;
            }
            else
            {
                index_1 = 1;
                index_4 = 0;
            }

            if (thePoints[3].Y > thePoints[2].Y)
            {
                index_2 = 2;
                index_3 = 3;
            }
            else
            {
                index_2 = 3;
                index_3 = 2;
            }

            box.Add(thePoints[index_1]);
            box.Add(thePoints[index_2]);
            box.Add(thePoints[index_3]);
            box.Add(thePoints[index_4]);

            return box;
        }

        public static int CompareByX(SKPoint left, SKPoint right)
        {
            /*
            if (left == null && right == null)
            {
                return 1;
            }

            if (left == null)
            {
                return 0;
            }

            if (right == null)
            {
                return 1;
            }
            */
            if (left.X > right.X)
            {
                return 1;
            }

            if (left.X == right.X)
            {
                return 0;
            }

            return -1;
        }

        private static double GetScore(SKPointI[] contours, SKImage fMapMat)
        {
            short xmin = 9999;
            short xmax = 0;
            short ymin = 9999;
            short ymax = 0;

            try
            {
                foreach (SKPointI point in contours)
                {
                    if (point.X < xmin)
                    {
                        xmin = (short)point.X;
                    }

                    if (point.X > xmax)
                    {
                        xmax = (short)point.X;
                    }

                    if (point.Y < ymin)
                    {
                        ymin = (short)point.Y;
                    }

                    if (point.Y > ymax)
                    {
                        ymax = (short)point.Y;
                    }
                }

                int roiWidth = xmax - xmin + 1;
                int roiHeight = ymax - ymin + 1;

                var gray8 = new SKImageInfo()
                {
                    Height = roiHeight,
                    Width = roiWidth,
                    AlphaType = SKAlphaType.Opaque,
                    ColorType = SKColorType.Gray8
                };

                byte[] roiBitmapSkBytes = new byte[gray8.BytesSize];

                using (SKImage roiBitmapSk = fMapMat.Subset(new SKRectI(xmin, ymin, xmax, ymax)))
                {
                    IntPtr buffer = Marshal.AllocHGlobal(gray8.BytesSize);
                    try
                    {
                        roiBitmapSk.ReadPixels(gray8, buffer);
                        Marshal.Copy(buffer, roiBitmapSkBytes, 0, gray8.BytesSize);
                    }
                    finally
                    {
                        Marshal.FreeHGlobal(buffer);
                    }
                }

                long sum = 0;
                int count = 0;

                using (SKBitmap mask = new SKBitmap(gray8))
                using (SKCanvas canvas = new SKCanvas(mask))
                using (SKPaint maskPaint = new SKPaint() { Color = SKColors.White, Style = SKPaintStyle.Fill })
                {
                    canvas.Clear(SKColors.Black);

                    var path = new SKPath();

                    SKPointI first = contours[0];
                    path.MoveTo(first.X - xmin, first.Y - ymin);
                    for (int p = 1; p < contours.Length; p++)
                    {
                        SKPointI point = contours[p];
                        path.LineTo(point.X - xmin, point.Y - ymin);
                    }
                    path.Close();

                    canvas.DrawPath(path, maskPaint);

                    for (int i = 0; i < mask.ByteCount; i++)
                    {
                        if (mask.Bytes[i] == byte.MaxValue)
                        {
                            sum += roiBitmapSkBytes[i];
                            count++;
                        }
                    }

                    path.Dispose();
                }

                if (count == 0)
                {
                    return 0;
                }
                return sum / (double)count / byte.MaxValue;
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine(ex.Message + ex.StackTrace);
            }

            return 0;
        }

        private static List<SKPointI> Unclip(List<SKPoint> box, float unclipRatio)
        {
            var boxArr = box.ToArray();
            SKPoint[] points = GeometryExtensionsF.MinimumAreaRectangle(boxArr);
            var size = GeometryExtensions.GetSize(points);

            if (size.height < 1.001 && size.width < 1.001)
            {
                return null;
            }

            var theClipperPts = new Path64(box.Select(pt => new Point64((int)pt.X, (int)pt.Y)));

            float area = Math.Abs(SignedPolygonArea(boxArr));
            double length = LengthOfPoints(box);
            double distance = area * unclipRatio / length;

            var co = new ClipperOffset();
            co.AddPath(theClipperPts, JoinType.Round, EndType.Polygon);
            var solution = new Paths64();
            co.Execute(distance, solution);
            if (solution.Count == 0)
            {
                return null;
            }

            List<SKPointI> retPts = new List<SKPointI>();
            foreach (var ip in solution[0])
            {
                retPts.Add(new SKPointI((int)ip.X, (int)ip.Y));
            }

            return retPts;
        }

        private static float SignedPolygonArea(SKPoint[] points)
        {
            // TODO - In place

            // Add the first point to the end.
            int numPoints = points.Length;
            SKPoint[] pts = new SKPoint[numPoints + 1];
            points.CopyTo(pts, 0);
            pts[numPoints] = points[0];

            // Get the areas.
            float area = 0;
            for (int i = 0; i < numPoints; i++)
            {
                area +=
                    (pts[i + 1].X - pts[i].X) *
                    (pts[i + 1].Y + pts[i].Y) / 2;
            }

            return area;
        }

        private static double LengthOfPoints(List<SKPoint> box)
        {
            double length = 0;

            SKPoint pt = box[0];
            double x0 = pt.X;
            double y0 = pt.Y;
            double x1 = 0, y1 = 0, dx = 0, dy = 0;
            box.Add(pt);

            int count = box.Count;
            for (int idx = 1; idx < count; idx++)
            {
                SKPoint pts = box[idx];
                x1 = pts.X;
                y1 = pts.Y;
                dx = x1 - x0;
                dy = y1 - y0;

                length += Math.Sqrt(dx * dx + dy * dy);

                x0 = x1;
                y0 = y1;
            }

            box.RemoveAt(count - 1);
            return length;
        }
    }
}
