using System;
using SkiaSharp;

namespace OcrLiteLib
{
    public sealed class ScaleParam
    {
        public int SrcWidth { get; set; }

        public int SrcHeight { get; set; }

        public int DstWidth { get; set; }

        public int DstHeight { get; set; }

        public float ScaleWidth { get; set; }

        public float ScaleHeight { get; set; }

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
            return $"sw:{this.SrcWidth},sh:{this.SrcHeight},dw:{this.DstWidth},dh:{this.DstHeight},{this.ScaleWidth},{this.ScaleHeight}";
        }

        public static ScaleParam GetScaleParam(SKBitmap src, int dstSize)
        {
            int srcWidth, srcHeight, dstWidth, dstHeight;
            srcWidth = src.Width;
            dstWidth = src.Width;
            srcHeight = src.Height;
            dstHeight = src.Height;

            float scale = 1.0F;
            if (dstWidth > dstHeight)
            {
                scale = dstSize / (float)dstWidth;
                dstWidth = dstSize;
                dstHeight = (int)(dstHeight * scale);
            }
            else
            {
                scale = dstSize / (float)dstHeight;
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