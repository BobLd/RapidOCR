using SkiaSharp;

namespace RapidOcrNet
{
    public readonly struct ScaleParam
    {
        public int SrcWidth { get; }

        public int SrcHeight { get; }

        public int DstWidth { get; }

        public int DstHeight { get; }

        public float ScaleWidth { get; }

        public float ScaleHeight { get; }

        public ScaleParam(int srcWidth, int srcHeight, int dstWidth, int dstHeight, float scaleWidth, float scaleHeight)
        {
            SrcWidth = srcWidth;
            SrcHeight = srcHeight;
            DstWidth = dstWidth;
            DstHeight = dstHeight;
            ScaleWidth = scaleWidth;
            ScaleHeight = scaleHeight;
        }

        public override string ToString()
        {
            return $"sw:{SrcWidth},sh:{SrcHeight},dw:{DstWidth},dh:{DstHeight},{ScaleWidth},{ScaleHeight}";
        }

        public static ScaleParam GetScaleParam(SKBitmap src, int dstSize)
        {
            int srcWidth = src.Width;
            int dstWidth = src.Width;
            int srcHeight = src.Height;
            int dstHeight = src.Height;

            if (dstWidth > dstHeight)
            {
                float scale = dstSize / (float)dstWidth;
                dstWidth = dstSize;
                dstHeight = (int)(dstHeight * scale);
            }
            else
            {
                float scale = dstSize / (float)dstHeight;
                dstHeight = dstSize;
                dstWidth = (int)(dstWidth * scale);
            }

            if (dstWidth % 32 != 0)
            {
                dstWidth = (dstWidth / 32 - 1) * 32;
                dstWidth = Math.Max(dstWidth, 32);
            }

            if (dstHeight % 32 != 0)
            {
                dstHeight = (dstHeight / 32 - 1) * 32;
                dstHeight = Math.Max(dstHeight, 32);
            }

            float scaleWidth = dstWidth / (float)srcWidth;
            float scaleHeight = dstHeight / (float)srcHeight;
            return new ScaleParam(srcWidth, srcHeight, dstWidth, dstHeight, scaleWidth, scaleHeight);
        }
    }
}