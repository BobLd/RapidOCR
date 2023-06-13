using SkiaSharp;

namespace OcrLib7Console
{
    public static class ContourHelper
    {
        private sealed record RowRun
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

        private sealed record Blob
        {
            public readonly List<RowRun> RowRuns = new List<RowRun>();

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

                RowRuns.Add(rowRun);
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

                RowRuns.AddRange(blob.RowRuns);
            }

            public bool IsOverlap(RowRun other)
            {
                return this.StartColumn <= other.End && other.Start <= this.EndColumn;
            }

            public bool IsOverlap(Blob other)
            {
                return this.StartColumn <= other.EndColumn && other.StartColumn <= this.EndColumn;
            }
        }

        private sealed class ContourExtractor
        {
            private readonly int totalCols;
            private readonly List<RowRun> currentRowRowRuns = new List<RowRun>();
            private readonly List<Blob> blobs = new List<Blob>();

            private RowRun? current;
            private int currentRow;
            private byte previousByte;

            public IReadOnlyList<Blob> Blobs => blobs;

            public ContourExtractor(int cols)
            {
                totalCols = cols;
            }

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

        public static SKPointI[][] FindContours(nint buffer, int rows, int cols)
        {
            return FindContoursInternal(buffer, rows, cols).ToArray();
        }

        public static SKPointI[][] FindBoxes(nint buffer, int rows, int cols)
        {
            return FindContoursInternal(buffer, rows, cols).Select(ParametricPerpendicularProjection).ToArray();
        }

        public static SKPointI[][] FindContours(byte[] array, int rows, int cols)
        {
            return FindContoursInternal(array, rows, cols).ToArray();
        }

        public static SKPointI[][] FindBoxes(byte[] array, int rows, int cols)
        {
            return FindContoursInternal(array, rows, cols).Select(ParametricPerpendicularProjection).ToArray();
        }

        private static IEnumerable<SKPointI[]> FindContoursInternal(nint buffer, int rows, int cols)
        {
            ContourExtractor manager = new ContourExtractor(cols);
            unsafe
            {
                byte* basePtr = (byte*)buffer.ToPointer();
                for (int r = 0; r < rows - 1; r++)
                {
                    manager.NextRow(r);
                    int rowOffset = r * cols;
                    for (int c = 0; c < cols - 1; c++)
                    {
                        int offset = rowOffset + c;
                        byte* ptr = basePtr + offset;
                        manager.ProcessNextByte(*ptr, c);
                    }
                }
            }

            return ProcessBlobs(manager.Blobs);
        }

        private static IEnumerable<SKPointI[]> FindContoursInternal(byte[] array, int rows, int cols)
        {
            var manager = new ContourExtractor(cols);
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

            return ProcessBlobs(manager.Blobs);
        }

        private static IEnumerable<SKPointI[]> ProcessBlobs(IReadOnlyList<Blob> blobs)
        {
            SKPointI[][] contours = new SKPointI[blobs.Count][];
            for (int b = 0; b < blobs.Count; b++)
            {
                HashSet<SKPointI> points = new HashSet<SKPointI>();
                foreach (var run in blobs[b].RowRuns.OrderBy(r => r.Row)) // TODO - check if order by useful
                {
                    if (run.Start == run.End)
                    {
                        points.Add(new SKPointI(run.Start, run.Row));
                    }
                    else
                    {
                        points.Add(new SKPointI(run.Start, run.Row));
                        points.Add(new SKPointI(run.End, run.Row));
                    }
                }
                contours[b] = points.ToArray();
            }

            return contours.Where(g => g.Length > 3)
                           .Select(g => GrahamScan(g).ToArray())
                           .Where(gr => gr.Length > 3);
                           //.OrderBy(gr => gr[0].X)
                           //.ThenByDescending(gr => gr[0].Y);
        }

        /// <summary>
        /// Algorithm to find the convex hull of the set of points with time complexity O(n log n).
        /// </summary>
        public static IEnumerable<SKPointI> GrahamScan(IEnumerable<SKPointI> points)
        {
            if (points?.Any() != true)
            {
                throw new ArgumentException("GrahamScan(): points cannot be null and must contain at least one point.", nameof(points));
            }

            if (points.Count() < 3) return points;

            static double polarAngle(SKPointI point1, SKPointI point2)
            {
                // This is used for grouping, we could use Math.Round()
                return Math.Atan2(point2.Y - point1.Y, point2.X - point1.X) % Math.PI;
            }

            var stack = new Stack<SKPointI>();
            var sortedPoints = points.OrderBy(p => p.X).ThenBy(p => p.Y).ToList();
            var P0 = sortedPoints[0];
            var groups = sortedPoints.Skip(1).GroupBy(p => polarAngle(P0, p)).OrderBy(g => g.Key);

            sortedPoints = new List<SKPointI>(); // Carefull here
            foreach (var group in groups)
            {
                if (group.Count() == 1)
                {
                    sortedPoints.Add(group.Single());
                }
                else
                {
                    // if more than one point has the same angle, 
                    // remove all but the one that is farthest from P0
                    sortedPoints.Add(group.MaxBy(p =>
                    {
                        double dx = p.X - P0.X;
                        double dy = p.Y - P0.Y;
                        return dx * dx + dy * dy;
                    }));
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
        /// Algorithm to find a minimal bounding rectangle (MBR) such that the MBR corresponds to a rectangle
        /// with smallest possible area completely enclosing the polygon.
        /// <para>From 'A Fast Algorithm for Generating a Minimal Bounding Rectangle' by Lennert D. Den Boer.</para>
        /// </summary>
        /// <param name="polygon">
        /// Polygon P is assumed to be both simple and convex, and to contain no duplicate (coincident) vertices.
        /// The vertices of P are assumed to be in strict cyclic sequential order, either clockwise or
        /// counter-clockwise relative to the origin P0.
        /// </param>
        public static SKPointI[] ParametricPerpendicularProjection(IReadOnlyList<SKPointI> polygon)
        {
            if (polygon == null || polygon.Count == 0)
            {
                throw new ArgumentException("ParametricPerpendicularProjection(): polygon cannot be null and must contain at least one point.", nameof(polygon));
            }
            else if (polygon.Count == 1)
            {
                return new SKPointI[] { polygon[0], polygon[0] };
            }
            else if (polygon.Count == 2)
            {
                return new SKPointI[] { polygon[0], polygon[1] };
            }

            double[] MBR = new double[8];

            double Amin = double.PositiveInfinity;
            int j = 1;
            int k = 0;

            double QX = double.NaN;
            double QY = double.NaN;
            double R0X = double.NaN;
            double R0Y = double.NaN;
            double R1X = double.NaN;
            double R1Y = double.NaN;

            while (true)
            {
                SKPoint Pk = polygon[k];
                SKPoint Pj = polygon[j];

                double vX = Pj.X - Pk.X;
                double vY = Pj.Y - Pk.Y;
                double r = 1.0 / (vX * vX + vY * vY);

                double tmin = 1;
                double tmax = 0;
                double smax = 0;
                int l = -1;
                double uX;
                double uY;

                for (j = 0; j < polygon.Count; j++)
                {
                    Pj = polygon[j];
                    uX = Pj.X - Pk.X;
                    uY = Pj.Y - Pk.Y;
                    double t = (uX * vX + uY * vY) * r;

                    double PtX = t * vX + Pk.X;
                    double PtY = t * vY + Pk.Y;
                    uX = PtX - Pj.X;
                    uY = PtY - Pj.Y;

                    double s = uX * uX + uY * uY;

                    if (t < tmin)
                    {
                        tmin = t;
                        R0X = PtX;
                        R0Y = PtY;
                    }

                    if (t > tmax)
                    {
                        tmax = t;
                        R1X = PtX;
                        R1Y = PtY;
                    }

                    if (s > smax)
                    {
                        smax = s;
                        QX = PtX;
                        QY = PtY;
                        l = j;
                    }
                }

                if (l != -1)
                {
                    SKPoint Pl = polygon[l];
                    double PlMinusQX = Pl.X - QX;
                    double PlMinusQY = Pl.Y - QY;

                    double R2X = R1X + PlMinusQX;
                    double R2Y = R1Y + PlMinusQY;

                    double R3X = R0X + PlMinusQX;
                    double R3Y = R0Y + PlMinusQY;

                    uX = R1X - R0X;
                    uY = R1Y - R0Y;

                    double A = (uX * uX + uY * uY) * smax;

                    if (A < Amin)
                    {
                        Amin = A;
                        MBR = new[] { R0X, R0Y, R1X, R1Y, R2X, R2Y, R3X, R3Y };
                    }
                }

                k++;
                j = k + 1;

                if (j == polygon.Count) j = 0;
                if (k == polygon.Count) break;
            }

            return new SKPointI[]
            {
                new SKPointI((int)MBR[4], (int)MBR[5]),
                new SKPointI((int)MBR[6], (int)MBR[7]),
                new SKPointI((int)MBR[2], (int)MBR[3]),
                new SKPointI((int)MBR[0], (int)MBR[1])
            };
        }

        /// <summary>
        /// Return true if the points are in counter-clockwise order.
        /// </summary>
        /// <param name="point1">The first point.</param>
        /// <param name="point2">The second point.</param>
        /// <param name="point3">The third point.</param>
        private static bool ccw(SKPointI point1, SKPointI point2, SKPointI point3)
        {
            return (point2.X - point1.X) * (point3.Y - point1.Y) > (point2.Y - point1.Y) * (point3.X - point1.X);
        }
    }
}
