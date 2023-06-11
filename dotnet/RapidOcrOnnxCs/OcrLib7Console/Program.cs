using System.Diagnostics;
using System.Drawing;
using System.Runtime.InteropServices;
using Microsoft.ML;
using Microsoft.ML.Data;
using SkiaSharp;
using static Microsoft.ML.Transforms.Image.ImagePixelExtractingEstimator;
using static Microsoft.ML.Transforms.Image.ImageResizingEstimator;

namespace OcrLib7Console
{
    internal class Program
    {
        private const float boxThresh = 0.3f;
        private const int dilateRadius = 2;

        private const string modelPath = @"C:\Users\Bob\Document Layout Analysis\RapidOCR\RapidOCR\dotnet\RapidOcrOnnxCs\models\en_PP-OCRv3_det_infer.onnx";

        private const string imageTestPath = @"C:\Users\Bob\Document Layout Analysis\text samples\5090.FontNameList.1_raw.png";
        //private const string imageTestPath = @"C:\Users\Bob\Document Layout Analysis\text samples\1.1_raw.png";
        //private const string imageTestPath = @"C:\Users\Bob\Document Layout Analysis\text samples\68-1990-01_A.2_raw.png";
        //private const string imageTestPath = @"C:\Users\Bob\Document Layout Analysis\text samples\5090.FontNameList.2_raw.png";

        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext();

            // Define scoring pipeline
            var pipeline = mlContext.Transforms.ResizeImages(inputColumnName: "bitmap", outputColumnName: "x", imageWidth: DbNetBitmap.Size, imageHeight: DbNetBitmap.Size, resizing: ResizingKind.IsoPad)
                            .Append(mlContext.Transforms.ExtractPixels(outputColumnName: "x", scaleImage: 1f / 255f, orderOfExtraction: ColorsOrder.ABGR))
                            .Append(mlContext.Transforms.ApplyOnnxModel(shapeDictionary: new Dictionary<string, int[]>()
                            {
                                { "x", new[] { 1, 3, DbNetBitmap.Size, DbNetBitmap.Size } },
                                { "sigmoid_0.tmp_0", new[] { 1, 1, DbNetBitmap.Size, DbNetBitmap.Size } }
                            },
                            inputColumnNames: new[] { "x" },
                            outputColumnNames: new[] { "sigmoid_0.tmp_0" },
                            modelFile: modelPath,
                            recursionLimit: 100));
            //.Append(mlContext.Transforms.Concatenate("result", new string[] { "sigmoid_0.tmp_0", "sigmoid_0.tmp_0", "sigmoid_0.tmp_0", "sigmoid_0.tmp_0" }))
            //.Append(mlContext.Transforms.ConvertToImage(imageHeight: DbNetBitmap.Size, imageWidth: DbNetBitmap.Size,
            //        outputColumnName: "result", inputColumnName: "result", colorsPresent: ColorBits.All, orderOfColors: ColorsOrder.ARGB,
            //        interleavedColors: false,
            //        scaleImage: 255))
            //.Append(mlContext.Transforms.ConvertToGrayscale(outputColumnName: "result", inputColumnName: "result"));

            // Fit on empty list to obtain input data schema
            var model = pipeline.Fit(mlContext.Data.LoadFromEnumerable(new List<DbNetBitmap>()));

            // Create prediction engine
            var engine = mlContext.Model.CreatePredictionEngine<DbNetBitmap, DbNetResult>(model);

            using (var image = MLImage.CreateFromFile(imageTestPath))
            {
                var dbNetBitmap = new DbNetBitmap()
                {
                    Image = image
                };

                var sw = Stopwatch.StartNew();
                DbNetResult result = engine.Predict(dbNetBitmap);
                Console.WriteLine($"Processing done in {sw.ElapsedMilliseconds}ms");

                Point[][]? contours = GetTextBox(result);
            }
        }

        public static Point[][]? GetTextBox(DbNetResult result)
        {
            var sw = Stopwatch.StartNew();
            Point[][]? contours;

            byte[] rawPixels = new byte[result.Result.Length];
            for (int i = 0; i < result.Result.Length; i++)
            {
                // Thresolding
                rawPixels[i] = result.Result[i] > boxThresh ? byte.MaxValue : byte.MinValue;
            }

            // No need to scale back, we just need to un-pad
            SKImage skImage = SKImage.FromPixelCopy(new SKImageInfo()
            {
                Height = DbNetBitmap.Size,
                Width = DbNetBitmap.Size,
                AlphaType = SKAlphaType.Opaque,
                ColorType = SKColorType.Gray8
            }, rawPixels);

#if DEBUG
            using (var fs = new FileStream("debug.png", FileMode.Create))
            using (SKData d = skImage.Encode(SKEncodedImageFormat.Png, 100))
            {
                d.SaveTo(fs);
            }
#endif

            // Remove padding
            double scale = DbNetBitmap.Size / (double)Math.Max(result.OriginalWidth, result.OriginalHeight);

            int scaledWidth = Convert.ToInt32(result.OriginalWidth * scale);
            int scaledHeight = Convert.ToInt32(result.OriginalHeight * scale);

            SKRectI crop;
            if (result.OriginalWidth > result.OriginalHeight)
            {
                // Padding added to height
                int padding = (DbNetBitmap.Size - scaledHeight) / 2;
                crop = new SKRectI(0, padding, DbNetBitmap.Size, DbNetBitmap.Size - padding);
            }
            else
            {
                // Padding added to width
                int padding = (DbNetBitmap.Size - scaledWidth) / 2;
                crop = new SKRectI(padding, 0, DbNetBitmap.Size - padding, DbNetBitmap.Size);
            }

            // Dilate - (NB: croppedDilated is not Gray8 anymore)
            SKImage croppedDilatedSubset;
            using (var filter = SKImageFilter.CreateDilate(dilateRadius, dilateRadius))
            using (SKImage croppedDilated = skImage.ApplyImageFilter(filter, crop, crop, out SKRectI subset, out SKPointI offset))
            {
                croppedDilatedSubset = croppedDilated.Subset(subset);
            }
            skImage.Dispose();

            var gray8 = new SKImageInfo()
            {
                Height = scaledHeight,
                Width = scaledWidth,
                AlphaType = SKAlphaType.Opaque,
                ColorType = SKColorType.Gray8
            };

            nint _buffer = Marshal.AllocHGlobal(gray8.BytesSize);
            croppedDilatedSubset.ReadPixels(gray8, _buffer);
            croppedDilatedSubset.Dispose();

            using (var finalBmp = new SKBitmap())
            {
                finalBmp.InstallPixels(gray8, _buffer);

#if DEBUG
                using (var fs = new FileStream("debug_cropped_dilate_gray8.png", FileMode.Create))
                using (SKData d = finalBmp.Encode(SKEncodedImageFormat.Png, 100))
                {
                    d.SaveTo(fs);
                }
#endif

                contours = ContourHelper.FindContours(finalBmp.Bytes, finalBmp.Height, finalBmp.Width); // Todo use Span
            }
            Marshal.FreeHGlobal(_buffer);

            // TODO - Scale back to page size

            Console.WriteLine($"Post processing done in {sw.ElapsedMilliseconds}ms");

            return contours;
        }
    }
}