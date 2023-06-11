using System.Drawing;

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

        public static Point[][] FindContours(byte[] array, int rows, int cols)
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
                foreach (var run in manager.Blobs[b].RowRuns.OrderBy(r => r.Row))
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

            return envelops;
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
    }
}
