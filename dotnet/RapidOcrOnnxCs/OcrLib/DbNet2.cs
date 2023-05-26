using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text.RegularExpressions;
using ClipperLib;
using Emgu.CV;
using Emgu.CV.Cuda;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OcrLiteLib;

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
                Console.WriteLine(ex.Message + ex.StackTrace);
                throw ex;
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
                    var textBoxes = GetTextBoxes(resultsArray, srcResize.Rows, srcResize.Cols, scale, boxScoreThresh, boxThresh, unClipRatio);
                    return textBoxes;
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message + ex.StackTrace);
            }
            return null;
        }

        private class RowRun
        {
            public int Row { get; set; }

            public int Start { get; set; }

            public int End { get; private set; }

            public bool IsClosed { get; private set; }

            public void Close(int end)
            {
                End = end;
                IsClosed = true;
            }

            public bool IsOverlap(RowRun other)
            {
                if (this.Start > other.End || other.Start > this.End)
                {
                    return false;
                }
                return true;
            }
        }

        private class Blob
        {
            public readonly List<RowRun> rowRuns = new List<RowRun>();

            public int StartRow { get; private set; } = -1;

            public int EndRow { get; private set; } = -1;

            public int StartColumn { get; private set; } = int.MaxValue;

            public int EndColumn { get; private set; } = int.MinValue;

            public void Add(RowRun rowRun)
            {
                // check if can add

                if (StartRow == -1)
                {
                    StartRow = rowRun.Row;
                }

                EndRow = rowRun.Row;

                if (rowRun.Start < StartColumn)
                {
                    StartColumn = rowRun.Start;
                }

                if (rowRun.End > EndColumn)
                {
                    EndColumn = rowRun.End;
                }

                rowRuns.Add(rowRun);
            }

            public void Add(Blob blob)
            {
                if (blob.StartRow < StartRow)
                {
                    StartRow = blob.StartRow;
                }

                if (blob.EndRow > EndRow)
                {
                    EndRow = blob.EndRow;
                }

                if (blob.StartColumn < StartColumn)
                {
                    StartColumn = blob.StartColumn;
                }

                if (blob.EndColumn > EndColumn)
                {
                    EndColumn = blob.EndColumn;
                }

                rowRuns.AddRange(blob.rowRuns);
            }

            public bool IsOverlap(RowRun other)
            {
                if (this.StartColumn > other.End || other.Start > this.EndColumn)
                {
                    return false;
                }
                return true;
            }

            public bool IsOverlap(Blob other)
            {
                if (this.StartColumn > other.EndColumn || other.StartColumn > this.EndColumn)
                {
                    return false;
                }
                return true;
            }
        }

        private class ContourExtractor
        {
            private readonly int totalCols;
            private readonly List<RowRun> currentRowRowRuns = new List<RowRun>();
            private readonly List<Blob> blobs = new List<Blob>();
            private RowRun current;

            public IReadOnlyList<Blob> Blobs => blobs;

            public ContourExtractor(int cols)
            {
                totalCols = cols;
            }

            private int currentRow;

            private byte previousByte;
            public void ProcessNextByte(byte current, int c)
            {
                if (current == previousByte)
                {
                    return;
                }

                if (previousByte == byte.MinValue && current == byte.MaxValue)
                {
                    Open(c);
                }
                else if (previousByte == byte.MaxValue && current == byte.MinValue)
                {
                    Close(c);
                }

                previousByte = current;
            }

            public void NextRow(int r)
            {
                if (current != null)
                {
                    Close(totalCols);
                }

                if (currentRowRowRuns.Count == 0)
                {
                    currentRow = r;
                    return;
                }

                for (int rr = 0; rr < currentRowRowRuns.Count; rr++)
                {
                    RowRun rowRun = currentRowRowRuns[rr];
                    if (rowRun == null)
                    {
                        continue;
                    }

                    var overlapping = blobs.Select((bl, index) => (bl, index))
                        .Where(b => (b.bl.EndRow == rowRun.Row - 1 || b.bl.EndRow == rowRun.Row) && b.bl.IsOverlap(rowRun))
                        .Select(b => b.index).ToArray();

                    if (overlapping.Length > 0)
                    {
                        Blob pivot = blobs[overlapping[0]];
                        pivot.Add(rowRun);

                        for (int i = overlapping.Length - 1; i > 0; i--)
                        {
                            pivot.Add(blobs[overlapping[i]]);
                            blobs.RemoveAt(overlapping[i]);
                        }
                    }
                    else
                    {
                        var blob = new Blob();
                        blob.Add(rowRun);
                        blobs.Add(blob);
                    }
                }

                currentRowRowRuns.Clear();
                currentRow = r;
            }

            public void Open(int column)
            {
                if (current?.IsClosed != false) // Close or null
                {
                    current = new RowRun() { Row = currentRow, Start = column };
                }
                else
                {
                    // throw
                }
            }

            public void Close(int column)
            {
                current.Close(column - 1);
                currentRowRowRuns.Add(current);
                current = null;
            }
        }

        private static VectorOfVectorOfPoint FindContours(byte[] array, int rows, int cols)
        {
            ContourExtractor manager = new ContourExtractor(cols);
            for (int r = 0; r < rows - 1; r++)
            {
                manager.NextRow(r);
                int rowOffset = r * cols;
                for (int c = 0; c < cols - 1; c++)
                {
                    int offset = rowOffset + c;
                    manager.ProcessNextByte(array[offset], c);
                }
            }

            Point[][] contours = new Point[manager.Blobs.Count][];
            for (int b = 0; b < manager.Blobs.Count; b++)
            {
                HashSet<Point> points = new HashSet<Point>();
                foreach (var run in manager.Blobs[b].rowRuns.OrderBy(r => r.Row))
                {
                    if (run.Start == run.End)
                    {
                        points.Add(new Point(run.Start, run.Row));
                    }
                    else
                    {
                        points.Add(new Point(run.Start, run.Row));
                        points.Add(new Point(run.End, run.Row));
                    }
                }
                contours[b] = points.ToArray();
            }

            var envelops = contours.Where(g => g.Length > 3)
                           .Select(g => GrahamScan(g).ToArray())
                           .Where(gr => gr.Length > 3)
                           .OrderBy(gr => gr[0].X)
                           .ThenByDescending(gr => gr[0].Y)
                           .ToArray();

            return new VectorOfVectorOfPoint(envelops);
        }

        /// <summary>
        /// Algorithm to find the convex hull of the set of points with time complexity O(n log n).
        /// </summary>
        public static IEnumerable<Point> GrahamScan(IEnumerable<Point> points)
        {
            if (points?.Any() != true)
            {
                throw new ArgumentException("GrahamScan(): points cannot be null and must contain at least one point.", nameof(points));
            }

            if (points.Count() < 3) return points;

            double polarAngle(Point point1, Point point2)
            {
                // This is used for grouping, we could use Math.Round()
                return Math.Atan2(point2.Y - point1.Y, point2.X - point1.X) % Math.PI;
            }

            var stack = new Stack<Point>();
            var sortedPoints = points.OrderBy(p => p.X).ThenBy(p => p.Y).ToList();
            var P0 = sortedPoints[0];
            var groups = sortedPoints.Skip(1).GroupBy(p => polarAngle(P0, p)).OrderBy(g => g.Key);

            sortedPoints = new List<Point>();
            foreach (var group in groups)
            {
                if (group.Count() == 1)
                {
                    sortedPoints.Add(group.First());
                }
                else
                {
                    // if more than one point has the same angle, 
                    // remove all but the one that is farthest from P0
                    sortedPoints.Add(group.OrderByDescending(p =>
                    {
                        double dx = p.X - P0.X;
                        double dy = p.Y - P0.Y;
                        return dx * dx + dy * dy;
                    }).First());
                }
            }

            if (sortedPoints.Count < 2)
            {
                return new[] { P0, sortedPoints[0] };
            }

            stack.Push(P0);
            stack.Push(sortedPoints[0]);
            stack.Push(sortedPoints[1]);

            for (int i = 2; i < sortedPoints.Count; i++)
            {
                var point = sortedPoints[i];
                while (stack.Count > 1 && !ccw(stack.ElementAt(1), stack.Peek(), point))
                {
                    stack.Pop();
                }
                stack.Push(point);
            }

            return stack;
        }

        /// <summary>
        /// Return true if the points are in counter-clockwise order.
        /// </summary>
        /// <param name="point1">The first point.</param>
        /// <param name="point2">The second point.</param>
        /// <param name="point3">The third point.</param>
        private static bool ccw(Point point1, Point point2, Point point3)
        {
            return (point2.X - point1.X) * (point3.Y - point1.Y) > (point2.Y - point1.Y) * (point3.X - point1.X);
        }

        private static List<TextBox> GetTextBoxes(DisposableNamedOnnxValue[] outputTensor, int rows, int cols, ScaleParam s, float boxScoreThresh, float boxThresh, float unClipRatio)
        {
            float maxSideThresh = 3.0f;//长边门限
            List<TextBox> rsBoxes = new List<TextBox>();
            //-----Data preparation-----
            float[] predData = outputTensor[0].AsEnumerable<float>().ToArray();

            byte[] cbufData = new byte[predData.Length];
            byte[] thresholed = new byte[cbufData.Length];

            for (int i = 0; i < predData.Length; i++)
            {
                float data = predData[i];
                cbufData[i] = Convert.ToByte(data * 255);
                //-----boxThresh-----
                thresholed[i] = data > boxThresh ? byte.MaxValue : byte.MinValue;
            }

            Mat predMat = new Mat(rows, cols, DepthType.Cv32F, 1);
            predMat.SetTo(predData);

            Mat cbufMat = new Mat(rows, cols, DepthType.Cv8U, 1);
            cbufMat.SetTo(cbufData);

            //-----boxThresh-----
            Mat thresholdMat = new Mat(rows, cols, DepthType.Cv8U, 1);
            thresholdMat.SetTo(thresholed);

            Image<Bgr, Byte> imgethresholdMat = thresholdMat.Clone().ToImage<Bgr, Byte>();
            imgethresholdMat.Save("imgethresholdMat.bmp");

            //-----dilate-----
            Mat dilateMat = new Mat();
            Mat dilateElement = CvInvoke.GetStructuringElement(ElementShape.Rectangle, new Size(2, 2), new Point(-1, -1));
            CvInvoke.Dilate(thresholdMat, dilateMat, dilateElement, new Point(-1, -1), 1, BorderType.Default, new MCvScalar(128, 128, 128));

            Image<Bgr, Byte> imgeOrigenal = dilateMat.Clone().ToImage<Bgr, Byte>();
            imgeOrigenal.Save("dilateMat.bmp");

            VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint();
            CvInvoke.FindContours(dilateMat, contours, null, RetrType.List, ChainApproxMethod.ChainApproxSimple);

            //VectorOfVectorOfPoint contours = FindContours(dilateMat.GetData().Cast<byte>().ToArray(), rows, cols);

            for (int i = 0; i < contours.Size; i++)
            {
                if (contours[i].Size <= 2)
                {
                    continue;
                }
                float maxSide = 0;
                List<PointF> minBox = GetMiniBox(contours[i], out maxSide);
                if (maxSide < maxSideThresh)
                {
                    continue;
                }
                double score = GetScore(contours[i], predMat);
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
            VectorOfPoint vop = new VectorOfPoint();
            vop.Push(contours.ToArray<Point>());
            return GetMiniBox(vop, out minEdgeSize);
        }

        private static List<PointF> GetMiniBox(VectorOfPoint contours, out float minEdgeSize)
        {
            List<PointF> box = new List<PointF>();
            RotatedRect rrect = CvInvoke.MinAreaRect(contours);
            PointF[] points = CvInvoke.BoxPoints(rrect);
            minEdgeSize = Math.Min(rrect.Size.Width, rrect.Size.Height);

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

        private static double GetScore(VectorOfPoint contours, Mat fMapMat)
        {
            short xmin = 9999;
            short xmax = 0;
            short ymin = 9999;
            short ymax = 0;

            try
            {
                foreach (Point point in contours.ToArray())
                {
                    if (point.X < xmin)
                    {
                        //var xx = nd[point.X];
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

                Image<Gray, float> bitmap = fMapMat.ToImage<Gray, float>();
                Image<Gray, float> roiBitmap = new Image<Gray, float>(roiWidth, roiHeight);
                float[,,] dataFloat = bitmap.Data;
                float[,,] data = roiBitmap.Data;

                for (int j = ymin; j < ymin + roiHeight; j++)
                {
                    for (int i = xmin; i < xmin + roiWidth; i++)
                    {
                        try
                        {
                            data[j - ymin, i - xmin, 0] = dataFloat[j, i, 0];
                        }
                        catch (Exception ex2)
                        {
                            Console.WriteLine(ex2.Message);
                        }
                    }
                }

                Mat mask = Mat.Zeros(roiHeight, roiWidth, DepthType.Cv8U, 1);
                List<Point> pts = new List<Point>();
                foreach (Point point in contours.ToArray())
                {
                    pts.Add(new Point(point.X - xmin, point.Y - ymin));
                }

                using (VectorOfPoint vp = new VectorOfPoint(pts.ToArray<Point>()))
                using (VectorOfVectorOfPoint vvp = new VectorOfVectorOfPoint(vp))
                {
                    CvInvoke.FillPoly(mask, vvp, new MCvScalar(1));
                }

                return CvInvoke.Mean(roiBitmap, mask).V0;
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message + ex.StackTrace);
            }

            return 0;
        }

        private static List<Point> Unclip(List<PointF> box, float unclip_ratio)
        {
            RotatedRect clipRect = CvInvoke.MinAreaRect(box.ToArray());
            if (clipRect.Size.Height < 1.001 && clipRect.Size.Width < 1.001)
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
