using System.Diagnostics;
using System.Text;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;

namespace RapidOcrNet
{
    internal sealed class CrnnNet
    {
        private readonly float[] MeanValues = [127.5F, 127.5F, 127.5F];
        private readonly float[] NormValues = [1.0F / 127.5F, 1.0F / 127.5F, 1.0F / 127.5F];
        private const int CrnnDstHeight = 48;
        //private const int CrnnCols = 6625;

        private InferenceSession _crnnNet;
        private IReadOnlyList<string> _keys;
        private string _inputName; // private List<string> _inputNames;
        
        ~CrnnNet()
        {
            _crnnNet.Dispose();
        }

        public void InitModel(string path, string keysPath, int numThread)
        {
            try
            {
                var op = new SessionOptions
                {
                    GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
                    InterOpNumThreads = numThread,
                    IntraOpNumThreads = numThread
                };

                _crnnNet = new InferenceSession(path, op);
                _inputName = _crnnNet.InputMetadata.Keys.First(); // _inputNames = _crnnNet.InputMetadata.Keys.ToList();
                _keys = InitKeys(keysPath);
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine(ex.Message + ex.StackTrace);
                throw;
            }
        }

        private static IReadOnlyList<string> InitKeys(string path)
        {
            using (var sr = new StreamReader(path, Encoding.UTF8))
            {
                List<string> keys = ["#"];

                string line;
                while ((line = sr.ReadLine()) != null)
                {
                    keys.Add(line);
                }

                keys.Add(" ");
                System.Diagnostics.Debug.WriteLine($"keys Size = {keys.Count}");

                return keys;
            }
        }

        public List<TextLine> GetTextLines(IReadOnlyList<SKBitmap> partImgs)
        {
            var sw = new Stopwatch();
            List<TextLine> textLines = new List<TextLine>();
            foreach (var i in partImgs)
            {
                sw.Restart();
                var textLine = GetTextLine(i);
                textLine.Time = sw.ElapsedMilliseconds;
                textLines.Add(textLine);
            }
            
            return textLines;
        }

        private TextLine GetTextLine(SKBitmap src)
        {
            float scale = CrnnDstHeight / (float)src.Height;
            int dstWidth = (int)(src.Width * scale);

            Tensor<float> inputTensors;
            using (SKBitmap srcResize = src.Resize(new SKSizeI(dstWidth, CrnnDstHeight), SKFilterQuality.High))
            {
                inputTensors = OcrUtils.SubtractMeanNormalize(srcResize, MeanValues, NormValues);
            }
            
            IReadOnlyCollection<NamedOnnxValue> inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(_inputName, inputTensors)
            };

            try
            {
                using (IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = _crnnNet.Run(inputs))
                {
                    var result = results[0];
                    var dimensions = result.AsTensor<float>().Dimensions;
                    ReadOnlySpan<float> outputData = result.AsEnumerable<float>().ToArray();

                    return ScoreToTextLine(outputData, dimensions[1], dimensions[2]);
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine(ex.Message + ex.StackTrace);
                //throw ex;
            }

            return new TextLine();
        }

        private TextLine ScoreToTextLine(ReadOnlySpan<float> srcData, int h, int w)
        {
            StringBuilder sb = new StringBuilder();
            TextLine textLine = new TextLine();

            int lastIndex = 0;
            List<float> scores = new List<float>();

            for (int i = 0; i < h; i++)
            {
                int maxIndex = 0;
                float maxValue = -1000F;
                for (int j = 0; j < w; j++)
                {
                    int idx = i * w + j;
                    if (srcData[idx] > maxValue)
                    {
                        maxIndex = j;
                        maxValue = srcData[idx];
                    }
                }

                if (maxIndex > 0 && maxIndex < _keys.Count && !(i > 0 && maxIndex == lastIndex))
                {
                    scores.Add(maxValue);
                    sb.Append(_keys[maxIndex]);
                }

                lastIndex = maxIndex;
            }

            textLine.Text = sb.ToString();
            textLine.CharScores = scores;
            return textLine;
        }
    }
}