import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;

public class trafficSignClassifierAnalysis {
	private static int _NestedLevel = 10;
	private static int _BasicPixel = 2934;
	private static int _StartDistance = 350;
	private static int _AverageNum = 100;
<<<<<<< HEAD
	
	
	private static int[] _Weight;
	private static int[] _Resize;
	private static float[][] _Accuracy;
=======
	private static int _Speed = 70; // km/h
	private static int _Period = 100;
	private static int _Times = 500;
	private static String _InputFile = "input.txt";
	private static String _OutputFile = "output.txt";
	private static String _TimeInputFile = "time_in.txt";
	private static String _TimeOutputFile = "time_out.txt";
	
	private static boolean _CalculateAccuracy = true;
	private static boolean _CalculateTime = false;
	private static boolean _WriteFile = true;


	private static int[] _Weight;
	private static int[] _Resize;
	private static double[][] _Accuracy;
	private static double[] _InferenceTime;
>>>>>>> 8594bdd67a8a2747b6d6b5fbb8984d9282d32790

	public static void main(String[] args) {

		FileReader fr = null;
		FileWriter fw = null;
<<<<<<< HEAD
		
		BufferedReader br = null;
		BufferedWriter bw = null;
		
		try {
			fr = new FileReader("input.txt");
			br = new BufferedReader(fr);
			
			String a = br.readLine();
			String[] tempLine = a.split(" ");
			int resizeLength = tempLine.length;
			_Weight = new int[resizeLength];
			_Resize = new int[resizeLength];
			
			/* Weight: resize * resize */
			for (int i=0; i<resizeLength; i++) {
				_Resize[i] = Integer.parseInt(tempLine[i]);
				_Weight[i] =  _Resize[i] * _Resize[i];
			}
			
			_Accuracy = new float[_NestedLevel][resizeLength];
			
			for (int i=0; i<_NestedLevel; i++)
			{
				a = br.readLine();
				tempLine = a.split(" ");
				for (int j=0; j<resizeLength; j++)
				{
					_Accuracy[i][j] = Float.parseFloat(tempLine[j+7]);
				}
			}
			
			float[] get = getAccuracies(7f);
			for (int i=0; i<get.length; i++)
				System.out.print(get[i] + " ");
			

=======

		BufferedReader br = null;
		BufferedWriter bw = null;

		try {
			if (_CalculateAccuracy) {
				fr = new FileReader(_InputFile);
				br = new BufferedReader(fr);

				if (_WriteFile) {
					fw = new FileWriter(_OutputFile);
					bw = new BufferedWriter(fw);
				}

				String a = br.readLine();
				String[] tempLine = a.split(" ");
				int resizeLength = tempLine.length;
				_Weight = new int[resizeLength];
				_Resize = new int[resizeLength];

				/* Weight: resize * resize */
				for (int i=0; i<resizeLength; i++) {
					_Resize[i] = Integer.parseInt(tempLine[i]);
					_Weight[i] =  _Resize[i] * _Resize[i];
				}

				_Accuracy = new double[_NestedLevel][resizeLength];

				for (int i=0; i<_NestedLevel; i++)
				{
					a = br.readLine();
					tempLine = a.split(" ");
					for (int j=0; j<resizeLength; j++)
					{
						_Accuracy[i][j] = Double.parseDouble(tempLine[j+7]);
					}
				}

				double basicSpeed = _Speed * 1000/3600.;
				double speed;
				double[] get;
				String result;
				for (int freq = 1; freq < _Times; freq++) // freq: _Times times per _Period sec
				{
					speed = basicSpeed / (double) freq * _Period;

					get = getAccuracies(speed);
					result = freq + " ";
					result += speed+" ";
					for (int i=0; i<get.length; i++)
						result += get[i] + " ";
					result += System.lineSeparator();
					System.out.print(result);
					if (_WriteFile)
						bw.write(result);
				}
			}
			
			if (_CalculateTime)
			{
				if (br != null) try {br.close(); br = null;} catch (Exception e) {}
				if (bw != null) try {bw.close(); bw = null;} catch (Exception e) {}
				if (fr != null) try {fr.close(); fr = null;} catch (Exception e) {}
				if (fw != null) try {fw.close(); fw = null;} catch (Exception e) {}
				fr = new FileReader(_TimeInputFile);
				br = new BufferedReader(fr);

				if (_WriteFile) {
					fw = new FileWriter(_TimeOutputFile);
					bw = new BufferedWriter(fw);
				}

				String a = br.readLine();
				String[] tempLine = a.split("\t");
				_InferenceTime = new double[tempLine.length];
				
				for (int i=0; i<tempLine.length; i++)
					_InferenceTime[i] = Double.parseDouble(tempLine[i]);
				
				double basicSpeed = _Speed * 1000/3600.;
				double speed;
				String result;
				
				for (int freq = 1; freq < _Times; freq++) // freq: _Times times per _Period sec
				{
					speed = basicSpeed / (double) freq * _Period;

					result = freq + " ";
					result += speed+" ";
					for (int i=0; i<_InferenceTime.length; i++)
						result += _InferenceTime[i] * freq / _Period + " ";
					result += System.lineSeparator();
					System.out.print(result);
					if (_WriteFile)
						bw.write(result);
				}
			}
>>>>>>> 8594bdd67a8a2747b6d6b5fbb8984d9282d32790
		} catch (Exception e) {
			e.printStackTrace();
		} finally {
			if (br != null) try {br.close(); br = null;} catch (Exception e) {}
			if (bw != null) try {bw.close(); bw = null;} catch (Exception e) {}
			if (fr != null) try {fr.close(); fr = null;} catch (Exception e) {}
			if (fw != null) try {fw.close(); fw = null;} catch (Exception e) {}
<<<<<<< HEAD
			
		}
	}
	
	static int getPixel(float distance)
	{
		int result = (int)(_BasicPixel / distance);
		return (result > 32) ? 32 : result;
	}
	
	static int getPostPixel(int pixel)
	{		
		int temp = -1;
	
=======

		}
	}

	static int getPixel(double distance)
	{
		if (distance < 0)
			System.out.println("ERROR, Wrong distance (getPixel)");
		int result = (int)(_BasicPixel / distance);
		return (result > 32) ? 32 : result;
	}

	static int getPostPixel(int pixel)
	{		
		int temp = -1;

>>>>>>> 8594bdd67a8a2747b6d6b5fbb8984d9282d32790
		for (int i=0; i<_Resize.length; i++) {
			if (_Resize[i] > pixel) {
				temp = i-1;
				break;
			}
		}
		if (pixel >= _Resize[_Resize.length-1])
			temp = _Resize.length-1;

<<<<<<< HEAD
		if (temp < 0)
			System.out.println("ERROR, check value temp");
		
		return _Resize[temp]; 
	}
	
	static float getAccuracy(int level, int pixel)
	{
		int temp = -1;
		
=======
		if (temp < 0) {
			System.out.println("ERROR, check value temp (getPostPixel)");
			return -1;
		}

		return _Resize[temp]; 
	}

	static double getAccuracy(int level, int pixel)
	{
		int temp = -1;

>>>>>>> 8594bdd67a8a2747b6d6b5fbb8984d9282d32790
		for (int i=0; i<_Resize.length; i++) {
			if (_Resize[i] > pixel) {
				temp = i-1;
				break;
			}
		}
<<<<<<< HEAD
		
		if (pixel >= _Resize[_Resize.length-1])
			temp = _Resize.length-1;
		
		if (temp < 0)
			System.out.println("ERROR, check value temp");
		
		return _Accuracy[level][temp];
	}
	
	static float[] getAccuracies(float speed)
	{
		//int n = (int)(_StartDistance / speed) + 1;
		float[] ret = new float[_NestedLevel];
		float interval = (speed) / _AverageNum; 
		
		for (int i=0; i < _NestedLevel; i++)
		{
			float sum = 0;
			for (int j=0; j < _AverageNum; j++)
			{
				float initialDistance = _StartDistance - j * interval;
				int scoreMax = 0;
				
				int n = (int)(initialDistance / speed);
				for (int k=0; k < n; k++)
				{
					float tempDistance = initialDistance - k * speed;
=======

		if (pixel >= _Resize[_Resize.length-1])
			temp = _Resize.length-1;

		if (temp < 0)
			System.out.println("ERROR, check value temp (getAccuracy)");

		return _Accuracy[level][temp];
	}

	static double[] getAccuracies(double speed)
	{
		//int n = (int)(_StartDistance / speed) + 1;
		double[] ret = new double[_NestedLevel];
		double interval = speed / _AverageNum; 

		for (int i=0; i < _NestedLevel; i++)
		{
			double sum = 0;
			int j;
			for (j=0; j < _AverageNum; j++)
			{
				double initialDistance = _StartDistance - j * interval;
				if (initialDistance < 0)
					continue;
				int scoreMax = 0;

				int n = (int)(initialDistance / speed) + 1;
				for (int k=0; k < n; k++)
				{
					double tempDistance = initialDistance - k * speed;
>>>>>>> 8594bdd67a8a2747b6d6b5fbb8984d9282d32790
					if (tempDistance < 0)
						break;
					int pixel = getPixel(tempDistance);
					int postPixel = getPostPixel(pixel);
					scoreMax += postPixel * postPixel;
				}
<<<<<<< HEAD
				
				float[][] probOfNScore = new float[n][scoreMax + 1];
=======

				double[][] probOfNScore = new double[n][scoreMax + 1];
>>>>>>> 8594bdd67a8a2747b6d6b5fbb8984d9282d32790
				{

					int pixel = getPixel(initialDistance);
					int postPixel = getPostPixel(pixel);
					int score = postPixel * postPixel;
<<<<<<< HEAD
					float accuracy = getAccuracy(i, postPixel);
=======
					double accuracy = getAccuracy(i, postPixel);
>>>>>>> 8594bdd67a8a2747b6d6b5fbb8984d9282d32790
					for (int jj=0; jj<scoreMax + 1; jj++)
					{
						if (jj == 0)
							probOfNScore[0][jj] = 1;
						else if (jj <= score)
							probOfNScore[0][jj] = accuracy;
						else
							probOfNScore[0][jj] = 0;
					}
				}
<<<<<<< HEAD
				for (int ii = 1; ii < n; ii++)
				{
					float tempDistance = initialDistance - ii * speed;
					if (tempDistance < 0)
						break;
					int pixel = getPixel(tempDistance);
					int postPixel = getPostPixel(pixel);
					int score = postPixel * postPixel;
					float accuracy = getAccuracy(i, postPixel);
					
=======
				int ii;
				for (ii = 1; ii < n; ii++)
				{
					double tempDistance = initialDistance - ii * speed;
					if (tempDistance < 0)
					{
						break;
					}
					int pixel = getPixel(tempDistance);
					int postPixel = getPostPixel(pixel);
					int score = postPixel * postPixel;
					double accuracy = getAccuracy(i, postPixel);

>>>>>>> 8594bdd67a8a2747b6d6b5fbb8984d9282d32790
					for (int jj=0; jj<scoreMax + 1; jj++)
					{
						probOfNScore[ii][jj] = probOfNScore[ii-1][jj] * (1-accuracy);
						if (jj > score) 
							probOfNScore[ii][jj] += probOfNScore[ii-1][jj-score] * accuracy;
						else
							probOfNScore[ii][jj] += accuracy;
					}
				}
<<<<<<< HEAD
				
				sum += probOfNScore[n-1][scoreMax / 2 + 1];
			}
			
			ret[i] = sum / _AverageNum;
		}
		
		return ret;
	}

=======
				ii--;

				sum += probOfNScore[ii][scoreMax / 2 + 1];
			}

			ret[i] = sum / _AverageNum;
		}

		return ret;
	}
	
>>>>>>> 8594bdd67a8a2747b6d6b5fbb8984d9282d32790
}
