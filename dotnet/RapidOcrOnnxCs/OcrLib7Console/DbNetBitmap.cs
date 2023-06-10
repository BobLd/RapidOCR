using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;

namespace OcrLib7Console
{
    public sealed class DbNetBitmap
    {
        public const int Size = 960; // Not all sizes seem to work, multiples of 64 apparently ok

        [ColumnName("bitmap")]
        [ImageType(Size, Size)]
        public MLImage Image { get; set; }

        [ColumnName("width")]
        public int ImageWidth => Image.Width;

        [ColumnName("height")]
        public int ImageHeight => Image.Height;
    }
}
