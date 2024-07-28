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
            List<Angle> angles = new List<Angle>();
            if (doAngle)
            {
                for (int i = 0; i < partImgs.Count; i++)
                {
                    var startTicks = DateTime.Now.Ticks;
                    var angle = GetAngle(partImgs[i]);
                    var endTicks = DateTime.Now.Ticks;
                    angle.Time = (endTicks - startTicks) / 10000F;
                    angles.Add(angle);
                }
            }
            else
            {
                for (int i = 0; i < partImgs.Count; i++)
                {
                    var angle = new Angle
                    {
                        Index = -1,
                        Score = 0F
                    };
                    angles.Add(angle);
                }
            }

            // Most Possible AngleIndex
            if (doAngle && mostAngle)
            {
                List<int> angleIndexes = new List<int>();
                angles.ForEach(x => angleIndexes.Add(x.Index));

                double sum = angleIndexes.Sum();
                double halfPercent = angles.Count / 2.0f;
                // All angles set to 0 or 1
                int mostAngleIndex = sum < halfPercent ? 0 : 1;
                System.Diagnostics.Debug.WriteLine($"Set All Angle to mostAngleIndex({mostAngleIndex})");
                for (int i = 0; i < angles.Count; ++i)
                {
                    angles[i].Index = mostAngleIndex;
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
                    var resultsArray = results.ToArray();
                    System.Diagnostics.Debug.WriteLine(resultsArray);
                    float[] outputData = resultsArray[0].AsEnumerable<float>().ToArray();
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

        private static Angle ScoreToAngle(float[] srcData, int angleCols)
        {
            int angleIndex = 0;
            float maxValue = -1000.0F;
            for (int i = 0; i < angleCols; i++)
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
