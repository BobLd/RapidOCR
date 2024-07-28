using System.Buffers;
using SkiaSharp;

namespace RapidOcrNet
{
    public static class GeometryExtensions
    {
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

        private sealed class PdfPointXYComparer : IComparer<SKPointI>
        {
            public static readonly PdfPointXYComparer Instance = new PdfPointXYComparer();

            public int Compare(SKPointI p1, SKPointI p2)
            {
                int comp = p1.X.CompareTo(p2.X);
                return comp == 0 ? p1.Y.CompareTo(p2.Y) : comp;
            }
        }

        private static double polarAngle(in SKPointI point1, in SKPointI point2)
        {
            // This is used for grouping, we could use Math.Round()
            return Math.Atan2(point2.Y - point1.Y, point2.X - point1.X) % Math.PI;
        }

        /// <summary>
        /// Algorithm to find the convex hull of the set of points with time complexity O(n log n).
        /// </summary>
        public static IReadOnlyCollection<SKPointI> GrahamScan(SKPointI[] points)
        {
            if (points is null || points.Length == 0)
            {
                throw new ArgumentException("GrahamScan(): points cannot be null and must contain at least one point.",
                    nameof(points));
            }

            if (points.Length < 3)
            {
                return points;
            }

            Array.Sort(points, PdfPointXYComparer.Instance);

            var P0 = points[0];
            var groups = points.Skip(1).GroupBy(p => polarAngle(P0, p)).OrderBy(g => g.Key).ToArray();

            var sortedPoints = ArrayPool<SKPointI>.Shared.Rent(groups.Length);

            try
            {
                for (int i = 0; i < groups.Length; i++)
                {
                    var group = groups[i];
                    if (group.Count() == 1)
                    {
                        sortedPoints[i] = group.First();
                    }
                    else
                    {
                        // if more than one point has the same angle, 
                        // remove all but the one that is farthest from P0
                        sortedPoints[i] = group.OrderByDescending(p =>
                        {
                            double dx = p.X - P0.X;
                            double dy = p.Y - P0.Y;
                            return dx * dx + dy * dy;
                        }).First();
                    }
                }

                if (groups.Length < 2)
                {
                    return new SKPointI[] { P0, sortedPoints[0] };
                }

                var stack = new Stack<SKPointI>();
                stack.Push(P0);
                stack.Push(sortedPoints[0]);
                stack.Push(sortedPoints[1]);

                for (int i = 2; i < groups.Length; i++)
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
            finally
            {
                ArrayPool<SKPointI>.Shared.Return(sortedPoints);
            }
        }

        /// <summary>
        /// Algorithm to find the (oriented) minimum area rectangle (MAR) by first finding the convex hull of the points
        /// and then finding its MAR.
        /// </summary>
        /// <param name="points">The points.</param>
        public static SKPoint[] MinimumAreaRectangle(SKPointI[] points)
        {
            if (points?.Any() != true)
            {
                throw new ArgumentException("MinimumAreaRectangle(): points cannot be null and must contain at least one point.", nameof(points));
            }

            return ParametricPerpendicularProjection(GrahamScan(points.Distinct().ToArray()).ToArray());
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
        private static SKPoint[] ParametricPerpendicularProjection(ReadOnlySpan<SKPointI> polygon)
        {
            if (polygon.Length == 0)
            {
                throw new ArgumentException("ParametricPerpendicularProjection(): polygon cannot be null and must contain at least one point.", nameof(polygon));
            }

            if (polygon.Length == 1)
            {
                return new SKPoint[] { polygon[0], polygon[0] };
            }

            if (polygon.Length == 2)
            {
                return new SKPoint[] { polygon[0], polygon[1] };
            }

            Span<float> mrb = stackalloc float[8];

            float Amin = float.PositiveInfinity;
            int j = 1;
            int k = 0;

            float QX = float.NaN;
            float QY = float.NaN;
            float R0X = float.NaN;
            float R0Y = float.NaN;
            float R1X = float.NaN;
            float R1Y = float.NaN;

            while (true)
            {
                SKPointI Pk = polygon[k];
                SKPointI Pj = polygon[j];

                int vX = Pj.X - Pk.X;
                int vY = Pj.Y - Pk.Y;
                float r = 1.0f / (vX * vX + vY * vY);

                float tmin = 1;
                float tmax = 0;
                float smax = 0;
                int l = -1;
                float uX;
                float uY;

                for (j = 0; j < polygon.Length; j++)
                {
                    Pj = polygon[j];
                    uX = Pj.X - Pk.X;
                    uY = Pj.Y - Pk.Y;
                    float t = (uX * vX + uY * vY) * r;

                    float PtX = t * vX + Pk.X;
                    float PtY = t * vY + Pk.Y;
                    uX = PtX - Pj.X;
                    uY = PtY - Pj.Y;

                    float s = uX * uX + uY * uY;

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
                    SKPointI Pl = polygon[l];
                    float PlMinusQX = Pl.X - QX;
                    float PlMinusQY = Pl.Y - QY;

                    float R2X = R1X + PlMinusQX;
                    float R2Y = R1Y + PlMinusQY;

                    float R3X = R0X + PlMinusQX;
                    float R3Y = R0Y + PlMinusQY;

                    uX = R1X - R0X;
                    uY = R1Y - R0Y;

                    float A = (uX * uX + uY * uY) * smax;

                    if (A < Amin)
                    {
                        Amin = A;

                        mrb[0] = R0X;
                        mrb[1] = R0Y;
                        mrb[2] = R1X;
                        mrb[3] = R1Y;
                        mrb[4] = R2X;
                        mrb[5] = R2Y;
                        mrb[6] = R3X;
                        mrb[7] = R3Y;
                    }
                }

                k++;
                j = k + 1;

                if (j == polygon.Length) j = 0;
                if (k == polygon.Length) break;
            }

            return new SKPoint[]
            {
                new SKPoint(mrb[4], mrb[5]),
                new SKPoint(mrb[6], mrb[7]),
                new SKPoint(mrb[2], mrb[3]),
                new SKPoint(mrb[0], mrb[1])
            };

            // RotatedRectangle(PointF topLeft, PointF topRight, PointF bottomLeft, PointF bottomRight)
        }

        public static (float width, float height) GetSize(SKPoint[] points)
        {
            SKPoint TopLeft = points[0];
            //PointF TopRight = points[1];
            SKPoint BottomLeft = points[2];
            SKPoint BottomRight = points[3];

            float t = 0;
            if (!BottomRight.Equals(BottomLeft))
            {
                t = MathF.Atan2(BottomRight.Y - BottomLeft.Y, BottomRight.X - BottomLeft.X);
            }
            else
            {
                // handle the case where both bottom points are identical
                t = MathF.Atan2(TopLeft.Y - BottomLeft.Y, TopLeft.X - BottomLeft.X) - MathF.PI / 2;
            }

            var cos = MathF.Cos(t);
            var sin = MathF.Sin(t);

            //Matrix inverseRotation = new Matrix(cos, -sin, sin, cos, 0, 0);
            //inverseRotation.TransformPoints(new SKPoint[] { BottomLeft, BottomRight, TopLeft });

            SKMatrix inverseRotation = new SKMatrix(cos, -sin, 0, sin, cos, 0, 0, 0, 1);
            BottomLeft = inverseRotation.MapPoint(BottomLeft);
            BottomRight = inverseRotation.MapPoint(BottomRight);
            TopLeft = inverseRotation.MapPoint(TopLeft);

            /*
            var inverseRotation = new TransformationMatrix(
                cos, -sin, 0,
                sin, cos, 0,
                0, 0, 1);
            */

            // Using Abs as a proxy for Euclidean distance in 1D 
            // as it might happen that points have negative coordinates.
            /*
            var bl = inverseRotation.Transform(BottomLeft);
            width = Math.Abs(inverseRotation.Transform(BottomRight).X - bl.X);
            height = Math.Abs(inverseRotation.Transform(TopLeft).Y - bl.Y);
            */

            var bl = BottomLeft;
            return (MathF.Abs(BottomRight.X - bl.X), MathF.Abs(TopLeft.Y - bl.Y));
        }
    }
}
