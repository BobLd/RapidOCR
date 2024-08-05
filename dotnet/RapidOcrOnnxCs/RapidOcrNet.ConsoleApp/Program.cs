﻿namespace RapidOcrNet.ConsoleApp
{
    internal class Program
    {
        private const string _modelsFolderName = "models";

        private static readonly int _numThreadNumeric = Environment.ProcessorCount;
        private static int padding = 50;
        private static int imgResize = 1024;
        private static float boxScoreThresh = 0.5f;
        private static float boxThresh = 0.3f;
        private static float unClipRatio = 1.6f;
        private static bool doAngle = true;
        private static bool mostAngle = false;

        static void Main(string[] args)
        {
            string targetImg = "C:\\Users\\Bob\\Document Layout Analysis\\text samples\\__test.png";

            Console.WriteLine("Hello, World!");
            string detPath = Path.Combine(_modelsFolderName, "en_PP-OCRv3_det_infer.onnx");
            string clsPath = Path.Combine(_modelsFolderName , "ch_ppocr_mobile_v2.0_cls_infer.onnx");
            string recPath = Path.Combine(_modelsFolderName, "en_PP-OCRv3_rec_infer.onnx");
            string keysPath = Path.Combine(_modelsFolderName, "en_dict.txt");

            if (!File.Exists(detPath))
            {
                throw new FileNotFoundException("Model file does not exist:" + detPath);
            }

            if (!File.Exists(clsPath))
            {
                throw new FileNotFoundException("Model file does not exist:" + clsPath);
            }

            if (!File.Exists(recPath))
            {
                throw new FileNotFoundException("Model file does not exist:" + recPath);
            }

            if (!File.Exists(recPath))
            {
                throw new FileNotFoundException("Keys file does not exist:" + keysPath);
            }

            var ocrEngin = new OcrLite();
            ocrEngin.InitModels(detPath, clsPath, recPath, keysPath, _numThreadNumeric);

            OcrResult ocrResult = ocrEngin.Detect(targetImg, padding, imgResize, boxScoreThresh, boxThresh, unClipRatio, doAngle, mostAngle);

            Console.WriteLine(ocrResult.ToString());
            Console.WriteLine(ocrResult.StrRes);
        }
    }
}
