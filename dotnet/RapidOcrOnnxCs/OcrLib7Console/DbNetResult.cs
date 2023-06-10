using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;

namespace OcrLib7Console
{
    public class DbNetResult
    {
        [VectorType(DbNetBitmap.Size * DbNetBitmap.Size)]
        [ColumnName("sigmoid_0.tmp_0")]
        public float[] Result { get; set; }

        [ColumnName("width")]
        public int OriginalWidth { get; set; }

        [ColumnName("height")]
        public int OriginalHeight { get; set; }

        //[ColumnName("result")]
        //[ImageType(DbNetBitmap.Size, DbNetBitmap.Size)]
        //public MLImage Image { get; set; }
    }
}
