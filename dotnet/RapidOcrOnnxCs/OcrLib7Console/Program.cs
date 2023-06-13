using System.Diagnostics;
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

        //private const string imageTestPath = @"C:\Users\Bob\Document Layout Analysis\text samples\5090.FontNameList.1_raw.png";
        private const string imageTestPath = @"C:\Users\Bob\Document Layout Analysis\text samples\1.1_raw.png";
        //private const string imageTestPath = @"C:\Users\Bob\Document Layout Analysis\text samples\68-1990-01_A.2_raw.png";
        //private const string imageTestPath = @"C:\Users\Bob\Document Layout Analysis\text samples\5090.FontNameList.2_raw.png";

        private static readonly float[] MeanValues = { 0.485F * 255F, 0.456F * 255F, 0.406F * 255F };
        private static readonly float[] NormValues = { 1.0F / 0.229F / 255.0F, 1.0F / 0.224F / 255.0F, 1.0F / 0.225F / 255.0F };

        static void Main(string[] args)
        {
            var sw = Stopwatch.StartNew();
            MLContext mlContext = new MLContext();

            // Define scoring pipeline
            var pipeline = mlContext.Transforms.ResizeImages(outputColumnName: "x", imageWidth: DbNetBitmap.Size, imageHeight: DbNetBitmap.Size, inputColumnName: "x", resizing: ResizingKind.IsoPad)
                            .Append(mlContext.Transforms.ExtractPixels(outputColumnName: "x", orderOfExtraction: ColorsOrder.ABGR, offsetImage: MeanValues.Average(), scaleImage: NormValues.Average())) // TODO - using average is not correct
                            .Append(mlContext.Transforms.ApplyOnnxModel(
                                outputColumnNames: new[] { "sigmoid_0.tmp_0" },
                                inputColumnNames: new[] { "x" },
                                modelFile: modelPath,
                                shapeDictionary: new Dictionary<string, int[]>()
                                {
                                    { "x", new[] { 1, 3, DbNetBitmap.Size, DbNetBitmap.Size } },
                                    { "sigmoid_0.tmp_0", new[] { 1, 1, DbNetBitmap.Size, DbNetBitmap.Size } }
                                },
                                recursionLimit: 100));
            Console.WriteLine($"Pipeline loaded in {sw.ElapsedMilliseconds}ms");

            sw.Restart();

            // Fit on empty list to obtain input data schema
            var model = pipeline.Fit(mlContext.Data.LoadFromEnumerable(new List<DbNetBitmap>()));
            Console.WriteLine($"Pipeline fitted in {sw.ElapsedMilliseconds}ms");

            sw.Restart();

            // Create prediction engine
            var engine = mlContext.Model.CreatePredictionEngine<DbNetBitmap, DbNetResult>(model);
            Console.WriteLine($"Prediction engine created in {sw.ElapsedMilliseconds}ms");

            using (var image = MLImage.CreateFromFile(imageTestPath))
            {
                var dbNetBitmap = new DbNetBitmap()
                {
                    Image = image
                };

                sw.Restart();
                DbNetResult result = engine.Predict(dbNetBitmap);
                Console.WriteLine($"Processing done in {sw.ElapsedMilliseconds}ms");

                SKPointI[][]? boxes = GetTextBox(result);
            }
        }

        public static SKPointI[][]? GetTextBox(DbNetResult result)
        {
            var sw = Stopwatch.StartNew();

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

            var gray8 = new SKImageInfo()
            {
                Height = DbNetBitmap.Size,
                Width = DbNetBitmap.Size,
                AlphaType = SKAlphaType.Opaque,
                ColorType = SKColorType.Gray8
            };

            var gray8Scaled = new SKImageInfo()
            {
                Height = scaledHeight,
                Width = scaledWidth,
                AlphaType = SKAlphaType.Opaque,
                ColorType = SKColorType.Gray8
            };

            byte[] rawPixels = new byte[result.Result.Length];
            for (int i = 0; i < result.Result.Length; i++)
            {
                // Thresolding
                rawPixels[i] = result.Result[i] > boxThresh ? byte.MaxValue : byte.MinValue;
            }

            // No need to scale back yet, we just need to un-pad

            nint buffer = Marshal.AllocHGlobal(gray8Scaled.BytesSize);

            using (var skImage = SKImage.FromPixelCopy(gray8, rawPixels)) // use pointer?
            using (var filter = SKImageFilter.CreateDilate(dilateRadius, dilateRadius))
            using (var croppedDilated = skImage.ApplyImageFilter(filter, crop, crop, out SKRectI subset, out SKPointI offset)) // Dilate
            using (var croppedDilatedSubset = croppedDilated.Subset(subset)) // Trim image due to dilate
            {
                // Store in buffer in Gray8 (NB: croppedDilated is not Gray8 anymore)
                croppedDilatedSubset.ReadPixels(gray8Scaled, buffer);
            }

            SKPointI[][]? contours = ContourHelper.FindBoxes(buffer, gray8Scaled.Height, gray8Scaled.Width);

#if DEBUG
            using (var finalBmp = new SKBitmap())
            {
                finalBmp.InstallPixels(gray8Scaled, buffer);
                using (var fs = new FileStream("debug_cropped_dilate_gray8.png", FileMode.Create))
                using (SKData d = finalBmp.Encode(SKEncodedImageFormat.Png, 100))
                {
                    d.SaveTo(fs);
                }
            }
#endif
            Marshal.FreeHGlobal(buffer);

            // TODO - get boxes

            // TODO - Scale back to page size

            Console.WriteLine($"Post processing done in {sw.ElapsedMilliseconds}ms");

            return contours;
        }
    }
}
