using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;

namespace RapidOcrNet
{
    internal sealed class AngleNet
    {
        private readonly float[] MeanValues = [127.5F, 127.5F, 127.5F];
        private readonly float[] NormValues = [1.0F / 127.5F, 1.0F / 127.5F, 1.0F / 127.5F];

        private const int angleDstWidth = 192;
        private const int angleDstHeight = 48;
        private const int angleCols = 2;

        private InferenceSession angleNet;
        private List<string> inputNames;

        public AngleNet()
        {
        }

        ~AngleNet()
        {
            angleNet.Dispose();
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
                angleNet = new InferenceSession(path, op);
                inputNames = angleNet.InputMetadata.Keys.ToList();
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
                foreach (var bmp in partImgs)
                {
                    var startTicks = DateTime.Now.Ticks;
                    var angle = GetAngle(bmp);
                    var endTicks = DateTime.Now.Ticks;
                    angle.Time = (endTicks - startTicks) / 10000F;
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
            using (var angleImg = src.Resize(new SKSizeI(angleDstWidth, angleDstHeight), SKFilterQuality.High))
            {
                inputTensors = OcrUtils.SubtractMeanNormalize(angleImg, MeanValues, NormValues);
            }

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(inputNames[0], inputTensors)
            };

            try
            {
                using (IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = angleNet.Run(inputs))
                {
                    ReadOnlySpan<float> outputData = results[0].AsEnumerable<float>().ToArray();
                    return ScoreToAngle(outputData, angleCols);
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
