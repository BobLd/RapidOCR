using Emgu.CV;
using Emgu.CV.CvEnum;
using OcrLiteLib;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace BaiPiaoOcrOnnxCs
{
    public partial class FormOcr : Form
    {
        private OcrLite ocrEngin;

        public FormOcr()
        {
            InitializeComponent();
        }

        private const string _modelsFolderName = "models";

        private void Form1_Load(object sender, EventArgs e)
        {
            string appPath = AppDomain.CurrentDomain.BaseDirectory;
            string rootDir = Directory.GetParent(appPath).Parent.Parent.Parent.Parent.FullName;
            string modelsDir = Path.Combine(rootDir, _modelsFolderName);
            if (!Directory.Exists(modelsDir))
            {
                modelsDir = Path.Combine(appPath, _modelsFolderName);
            }
            modelsTextBox.Text = modelsDir;
            string detPath = modelsDir + "\\" + detNameTextBox.Text;
            string clsPath = modelsDir + "\\" + clsNameTextBox.Text;
            string recPath = modelsDir + "\\" + recNameTextBox.Text;
            string keysPath = modelsDir + "\\" + keysNameTextBox.Text;
            bool isDetExists = File.Exists(detPath);
            if (!isDetExists)
            {
                MessageBox.Show("Model file does not exist:" + detPath); // 模型文件不存在
            }
            bool isClsExists = File.Exists(clsPath);
            if (!isClsExists)
            {
                MessageBox.Show("Model file does not exist:" + clsPath); // 模型文件不存在
            }
            bool isRecExists = File.Exists(recPath);
            if (!isRecExists)
            {
                MessageBox.Show("Model file does not exist:" + recPath); // 模型文件不存在
            }
            bool isKeysExists = File.Exists(recPath);
            if (!isKeysExists)
            {
                MessageBox.Show("Keys file does not exist:" + keysPath); // Keys文件不存在
            }
            if (isDetExists && isClsExists && isRecExists && isKeysExists)
            {
                ocrEngin = new OcrLite();
                ocrEngin.InitModels(detPath, clsPath, recPath, keysPath, (int)numThreadNumeric.Value);
            }
            else
            {
                MessageBox.Show("Initialization failed, please confirm the model folder and files, and re-initialize!"); // 初始化失败，请确认模型文件夹和文件后，重新初始化！
            }
        }

        // https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/deploy/lite/readme.md
        private void initBtn_Click(object sender, EventArgs e)
        {
            string modelsDir = modelsTextBox.Text;
            string detPath = modelsDir + "\\" + detNameTextBox.Text;
            string clsPath = modelsDir + "\\" + clsNameTextBox.Text;
            string recPath = modelsDir + "\\" + recNameTextBox.Text;
            string keysPath = modelsDir + "\\" + keysNameTextBox.Text;
            bool isDetExists = File.Exists(detPath);
            if (!isDetExists)
            {
                MessageBox.Show("Model file does not exist:" + detPath); // 模型文件不存在
            }
            bool isClsExists = File.Exists(clsPath);
            if (!isClsExists)
            {
                MessageBox.Show("Model file does not exist:" + clsPath); // 模型文件不存在
            }
            bool isRecExists = File.Exists(recPath);
            if (!isRecExists)
            {
                MessageBox.Show("Model file does not exist:" + recPath); // 模型文件不存在
            }
            bool isKeysExists = File.Exists(recPath);
            if (!isKeysExists)
            {
                MessageBox.Show("Keys file does not exist:" + keysPath); // Keys文件不存在
            }
            if (isDetExists && isClsExists && isRecExists && isKeysExists)
            {
                ocrEngin = new OcrLite();
                ocrEngin.InitModels(detPath, clsPath, recPath, keysPath, (int)numThreadNumeric.Value);
            }
            else
            {
                MessageBox.Show("Initialization failed, please confirm the model folder and files, and re-initialize!"); // 初始化失败，请确认模型文件夹和文件后，重新初始化
            }
        }

        private void openBtn_Click(object sender, EventArgs e)
        {
            using (var dlg = new OpenFileDialog())
            {
                dlg.Multiselect = false;
                dlg.Filter = "(*.JPG,*.PNG,*.JPEG,*.BMP,*.GIF)|*.JPG;*.PNG;*.JPEG;*.BMP;*.GIF|All files(*.*)|*.*";
                if (dlg.ShowDialog() == DialogResult.OK && !string.IsNullOrEmpty(dlg.FileName))
                {
                    pathTextBox.Text = dlg.FileName;
                    Mat src = CvInvoke.Imread(dlg.FileName, ImreadModes.Color);
                    pictureBox.Image = src.ToBitmap();
                }
            }
        }

        private void modelsBtn_Click(object sender, EventArgs e)
        {
            using (var dlg = new FolderBrowserDialog())
            {
                dlg.SelectedPath = Environment.CurrentDirectory + "\\models";
                if (dlg.ShowDialog() == DialogResult.OK && !string.IsNullOrEmpty(dlg.SelectedPath))
                {
                    modelsTextBox.Text = dlg.SelectedPath;
                }
            }
        }

        private void detectBtn_Click(object sender, EventArgs e)
        {
            if (ocrEngin == null)
            {
                MessageBox.Show("Uninitialized, cannot execute!"); // 未初始化，无法执行!
                return;
            }
            string targetImg = pathTextBox.Text;
            if (!File.Exists(targetImg))
            {
                MessageBox.Show("The target picture does not exist, please use the Open button to open"); // 目标图片不存在，请用Open按钮打开
                return;
            }
            int padding = (int)paddingNumeric.Value;
            int imgResize = (int)imgResizeNumeric.Value;
            float boxScoreThresh = (float)boxScoreThreshNumeric.Value;
            float boxThresh = (float)boxThreshNumeric.Value;
            float unClipRatio = (float)unClipRatioNumeric.Value;
            bool doAngle = doAngleCheckBox.Checked;
            bool mostAngle = mostAngleCheckBox.Checked;
            OcrResult ocrResult = ocrEngin.Detect(pathTextBox.Text, padding, imgResize, boxScoreThresh, boxThresh, unClipRatio, doAngle, mostAngle);
            ocrResultTextBox.Text = ocrResult.ToString();
            strRestTextBox.Text = ocrResult.StrRes;
            //pictureBox.Image = ocrResult.BoxImg.ToBitmap();
        }

        private void partImgCheckBox_CheckedChanged(object sender, EventArgs e)
        {
            ocrEngin.isPartImg = partImgCheckBox.Checked;
        }

        private void debugCheckBox_CheckedChanged(object sender, EventArgs e)
        {
            ocrEngin.isDebugImg = debugCheckBox.Checked;
        }
    }
}
