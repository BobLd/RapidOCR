using System.Diagnostics;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;

namespace RapidOcrNet
{
    internal sealed class AngleNet
    {
        private const int AngleDstWidth = 192;
        private const int AngleDstHeight = 48;
        private const int AngleCols = 2;

        private readonly float[] MeanValues = [127.5F, 127.5F, 127.5F];
        private readonly float[] NormValues = [1.0F / 127.5F, 1.0F / 127.5F, 1.0F / 127.5F];

        private InferenceSession _angleNet;
        private string _inputName; //private List<string> _inputNames;

        ~AngleNet()
        {
            _angleNet.Dispose();
        }

        public void InitModel(string path, int numThread)
        {
            try
            {
                var op = new SessionOptions
                {
                    GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
                    InterOpNumThreads = numThread,
                    IntraOpNumThreads = numThread
                };
                _angleNet = new InferenceSession(path, op);
                _inputName = _angleNet.InputMetadata.Keys.First(); //_inputNames = _angleNet.InputMetadata.Keys.ToList();
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine(ex.Message + ex.StackTrace);
                throw;
            }
        }

        public List<Angle> GetAngles(IReadOnlyList<SKBitmap> partImgs, bool doAngle, bool mostAngle)
        {
            var angles = new List<Angle>();
            if (doAngle)
            {
                var sw = new Stopwatch();

                foreach (var bmp in partImgs)
                {
                    sw.Restart();
                    var angle = GetAngle(bmp);
                    angle.Time = sw.ElapsedMilliseconds;
                    angles.Add(angle);
                }
            }
            else
            {
                for (int i = 0; i < partImgs.Count; i++)
                {
                    angles.Add(new Angle
                    {
                        Index = -1,
                        Score = 0F
                    });
                }
            }

            // Most Possible AngleIndex
            if (doAngle && mostAngle)
            {
                double sum = angles.Sum(x => x.Index);
                double halfPercent = angles.Count / 2.0f;

                int mostAngleIndex = sum < halfPercent ? 0 : 1; // All angles set to 0 or 1
                System.Diagnostics.Debug.WriteLine($"Set All Angle to mostAngleIndex({mostAngleIndex})");
                foreach (var angle in angles)
                {
                    angle.Index = mostAngleIndex;
                }
            }

            return angles;
        }

        private Angle GetAngle(SKBitmap src)
        {
            Tensor<float> inputTensors;
            using (var angleImg = src.Resize(new SKSizeI(AngleDstWidth, AngleDstHeight), SKFilterQuality.High))
            {
                inputTensors = OcrUtils.SubtractMeanNormalize(angleImg, MeanValues, NormValues);
            }

            IReadOnlyCollection<NamedOnnxValue> inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(_inputName, inputTensors)
            };

            try
            {
                using (IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = _angleNet.Run(inputs))
                {
                    ReadOnlySpan<float> outputData = results[0].AsEnumerable<float>().ToArray();
                    return ScoreToAngle(outputData, AngleCols);
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine(ex.Message + ex.StackTrace);
                //throw;
            }

            return new Angle();
        }

        private static Angle ScoreToAngle(ReadOnlySpan<float> srcData, int angleColumns)
        {
            int angleIndex = 0;
            float maxValue = -1000.0F;
            for (int i = 0; i < angleColumns; i++)
            {
                if (i == 0) maxValue = srcData[i];
                else if (srcData[i] > maxValue)
                {
                    angleIndex = i;
                    maxValue = srcData[i];
                }
            }

            return new Angle
            {
                Index = angleIndex,
                Score = maxValue
            };
        }
    }
}
