using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Runtime.InteropServices;
using ClipperLib;
using Emgu.CV;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OcrLib;
using PContourNet;
using SkiaSharp;

namespace OcrLiteLib
{
    internal class DbNet2
    {
        private readonly float[] MeanValues = { 0.485F * 255F, 0.456F * 255F, 0.406F * 255F };
        private readonly float[] NormValues = { 1.0F / 0.229F / 255.0F, 1.0F / 0.224F / 255.0F, 1.0F / 0.225F / 255.0F };

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
                SessionOptions op = new SessionOptions();
                op.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_EXTENDED;
                op.InterOpNumThreads = numThread;
                op.IntraOpNumThreads = numThread;
                dbNet = new InferenceSession(path, op);
                inputNames = dbNet.InputMetadata.Keys.ToList();
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine(ex.Message + ex.StackTrace);
                throw;
            }
        }

        public List<TextBox> GetTextBoxes(SKBitmap src, ScaleParam scale, float boxScoreThresh, float boxThresh, float unClipRatio)
        {
            using (var srcResize = src.Resize(new SKSizeI(scale.DstWidth, scale.DstHeight), SKFilterQuality.High))
            {
                Tensor<float> inputTensors = OcrUtils.SubstractMeanNormalize(srcResize, MeanValues, NormValues);
                var inputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor(inputNames[0], inputTensors)
                };
                try
                {
                    using (IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = dbNet.Run(inputs))
                    {
                        var resultsArray = results.ToArray();
                        Console.WriteLine(resultsArray);
                        return GetTextBoxes(resultsArray, srcResize.Height, srcResize.Width, scale, boxScoreThresh, boxThresh, unClipRatio);
                    }
                }
                catch (Exception ex)
                {
                    System.Diagnostics.Debug.WriteLine(ex.Message + ex.StackTrace);
                }
                return null;
            }
        }

        public List<TextBox> GetTextBoxes(Mat src, ScaleParam scale, float boxScoreThresh, float boxThresh, float unClipRatio)
        {
            Mat srcResize = new Mat();
            CvInvoke.Resize(src, srcResize, new Size(scale.DstWidth, scale.DstHeight));
            Tensor<float> inputTensors = OcrUtils.SubstractMeanNormalize(srcResize, MeanValues, NormValues);
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(inputNames[0], inputTensors)
            };
            try
            {
                using (IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = dbNet.Run(inputs))
                {
                    var resultsArray = results.ToArray();
                    Console.WriteLine(resultsArray);
                    return GetTextBoxes(resultsArray, srcResize.Rows, srcResize.Cols, scale, boxScoreThresh, boxThresh, unClipRatio);
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine(ex.Message + ex.StackTrace);
            }
            return null;
        }

        private static Point[][] FindContours(ReadOnlySpan<byte> array, int rows, int cols)
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

            SKImage predMatImage = SKImage.FromPixelCopy(gray8, predData.Select(b => Convert.ToByte(b * 255)).ToArray());

            var crop = new SKRectI(0, 0, cols, rows);

            Span<byte> rawPixels = new byte[predData.Length];
            for (int i = 0; i < predData.Length; i++)
            {
                // Thresholding
                rawPixels[i] = predData[i] > boxThresh ? byte.MaxValue : byte.MinValue;
            }

            const int dilateRadius = 2;

            Point[][] contours;
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

            for (int i = 0; i < contours.Length; i++)
            {
                if (contours[i].Length <= 2)
                {
                    continue;
                }

                float maxSide = 0;
                List<PointF> minBox = GetMiniBox(contours[i], out maxSide);
                if (maxSide < maxSideThresh)
                {
                    continue;
                }

                double score = GetScore(contours[i], predMatImage);
                if (score < boxScoreThresh)
                {
                    continue;
                }

                List<Point> clipBox = Unclip(minBox, unClipRatio);
                if (clipBox == null)
                {
                    continue;
                }

                List<PointF> clipMinBox = GetMiniBox(clipBox, out maxSide);
                if (maxSide < maxSideThresh + 2)
                {
                    continue;
                }

                List<Point> finalPoints = new List<Point>();
                foreach (var item in clipMinBox)
                {
                    int x = (int)(item.X / s.ScaleWidth);
                    int ptx = Math.Min(Math.Max(x, 0), s.SrcWidth);

                    int y = (int)(item.Y / s.ScaleHeight);
                    int pty = Math.Min(Math.Max(y, 0), s.SrcHeight);
                    Point dstPt = new Point(ptx, pty);
                    finalPoints.Add(dstPt);
                }

                TextBox textBox = new TextBox();
                textBox.Score = (float)score;
                textBox.Points = finalPoints;
                rsBoxes.Add(textBox);
            }

            rsBoxes.Reverse();
            return rsBoxes;
        }

        private static List<PointF> GetMiniBox(List<Point> contours, out float minEdgeSize)
        {
            return GetMiniBox(contours.ToArray(), out minEdgeSize);
        }

        private static List<PointF> GetMiniBox(Point[] contours, out float minEdgeSize)
        {
            List<PointF> box = new List<PointF>();

            PointF[] points = GeometryExtensions.MinimumAreaRectangle(contours);

            var size = GeometryExtensions.GetSize(points);
            minEdgeSize = Math.Min(size.width, size.height);

            List<PointF> thePoints = new List<PointF>(points);
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

        public static int CompareByX(PointF left, PointF right)
        {
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

        private static double GetScore(Point[] contours, SKImage fMapMat)
        {
            short xmin = 9999;
            short xmax = 0;
            short ymin = 9999;
            short ymax = 0;

            try
            {
                foreach (Point point in contours)
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

                    var points = contours.ToArray();
                    SKPath path = new SKPath();

                    Point first = points[0];
                    path.MoveTo(first.X - xmin, first.Y - ymin);
                    for (int p = 1; p < points.Length; p++)
                    {
                        Point point = points[p];
                        path.LineTo(point.X - xmin, point.Y - ymin);
                    }
                    path.Close();

                    canvas.DrawPath(path, maskPaint);

                    for (int i = 0; i < mask.ByteCount; i++)
                    {
                        if (mask.Bytes[i] == 255)
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

        private static List<Point> Unclip(List<PointF> box, float unclip_ratio)
        {
            PointF[] points = GeometryExtensionsF.MinimumAreaRectangle(box.ToArray());
            var size = GeometryExtensions.GetSize(points);

            if (size.height < 1.001 && size.width < 1.001)
            {
                return null;
            }

            List<IntPoint> theCliperPts = new List<IntPoint>();
            foreach (PointF pt in box)
            {
                IntPoint a1 = new IntPoint((int)pt.X, (int)pt.Y);
                theCliperPts.Add(a1);
            }

            float area = Math.Abs(SignedPolygonArea(box.ToArray<PointF>()));
            double length = LengthOfPoints(box);
            double distance = area * unclip_ratio / length;

            ClipperOffset co = new ClipperOffset();
            co.AddPath(theCliperPts, JoinType.jtRound, EndType.etClosedPolygon);
            List<List<IntPoint>> solution = new List<List<IntPoint>>();
            co.Execute(ref solution, distance);
            if (solution.Count == 0)
            {
                return null;
            }

            List<Point> retPts = new List<Point>();
            foreach (IntPoint ip in solution[0])
            {
                retPts.Add(new Point((int)ip.X, (int)ip.Y));
            }

            return retPts;
        }

        private static float SignedPolygonArea(PointF[] Points)
        {
            // Add the first point to the end.
            int num_points = Points.Length;
            PointF[] pts = new PointF[num_points + 1];
            Points.CopyTo(pts, 0);
            pts[num_points] = Points[0];

            // Get the areas.
            float area = 0;
            for (int i = 0; i < num_points; i++)
            {
                area +=
                    (pts[i + 1].X - pts[i].X) *
                    (pts[i + 1].Y + pts[i].Y) / 2;
            }

            return area;
        }

        private static double LengthOfPoints(List<PointF> box)
        {
            double length = 0;

            PointF pt = box[0];
            double x0 = pt.X;
            double y0 = pt.Y;
            double x1 = 0, y1 = 0, dx = 0, dy = 0;
            box.Add(pt);

            int count = box.Count;
            for (int idx = 1; idx < count; idx++)
            {
                PointF pts = box[idx];
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
