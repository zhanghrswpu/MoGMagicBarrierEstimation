package datamodel;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.io.RandomAccessFile;
import java.util.Properties;
import java.util.Random;

/**
 * <p>
 * Summary: Rating data information.
 * <p>
 * Author: <b>Henry</b> zhanghrswpu@163.com <br>
 * Copyright: The source code and all documents are open and free. PLEASE keep
 * this header while revising the program. <br>
 * Organization: <a href=http://www.fansmale.com/>Lab of Machine Learning</a>,
 * SouthWest Petroleum University, Sichuan 610500, China.<br>
 * Progress: OK. Copied from Hydrosimu.<br>
 * Written time: August 15, 2017. <br>
 * Last modify time: August 15, 2017.
 */
public class DataInfo {
	/**
	 * obtain the parameters from property file
	 */
	public Properties settings = new Properties();

	/**
	 * user number
	 */
	public int userNum;

	/**
	 * item number
	 */
	public int itemNum;

	/**
	 * number of ratings
	 */
	public int numOfRatings;

	/**
	 * triple rating data information (user, item, rating)
	 */
	public Triple[] trData;

	/**
	 * User set
	 */
	public float[][] uRatings; // The rating matrix for the user
	public int[][] uRateInds;// The rating indices for the user
	public int[] uDgr; // The degrees for the user
	
	/**
	 * item set
	 */
	public float[][] iRatings; // The rating matrix for the item
	public int[][] iRateInds;// The rating indices for the item
	public int[] iDgr; // The degrees for the item

	/**
	 * The mean/average value of rating for the data set. It is equal to
	 * sum(trainVector)/trainVector.length
	 */
	public float totalOfRatings;
	public float meanOfRatings;

	/**
	 * Store non-zero element of data set as a vector. It is a compressed version
	 * with out considering the position.
	 */
	public float[] dataVector;

	/**
	 * The small matrix U.
	 */
	public float[][] subU;

	/**
	 * The small matrix V. U * V approximate R (non-zero part).
	 */
	public float[][] subV;

	/********************** Feature Matrix ***********************************/
	/**
	 * The rank of small matrices. It is often 2 or 4. It is also the number of
	 * latent variable.
	 */
	public int rank;

	/**
	 * Matrix Factorization iteration times
	 */
	public int MFIterTimes;

	/**
	 * Convergence condition
	 */
	float convergence;

	/**
	 * Number of Gaussian noise.
	 */
	public int numNoise;

	/**
	 * The noise type of each rating. [0.2, 0.1, 0.7] indicates that the probability
	 * is 0.2 for the first noise.
	 */
	float[][] initialRatingsNoiseDistribution;

	/**
	 * The weight of every noise.
	 */
	public float[] noiseWeight;

	public String dataFile;
	public String splitStr;

	/**
	 * rating scale.
	 */
	public int LEVEL;
	public double rlow; // the lowest rating level
	public double rhigh; // the highest rating level
	public double stepLen; // the length of step

	/**
	 * max iteration times
	 */
	public int maxIterTimes;

	/**
	 * 
	 * @param paraFileName
	 * @throws Exception
	 */
	public DataInfo(String paraPropertyFilename) throws Exception {
		// Step 1. setting the parameters
		InputStream tempInputStream = new BufferedInputStream(new FileInputStream(paraPropertyFilename));
		settings.load(tempInputStream);
		userNum = Integer.parseInt(settings.getProperty("numUsers"));
		itemNum = Integer.parseInt(settings.getProperty("numItems"));
		numOfRatings = Integer.parseInt(settings.getProperty("numOfRatings"));
		rank = Short.parseShort(settings.getProperty("rank"));
		MFIterTimes = Integer.parseInt(settings.getProperty("matrixFactorizationIterTimes"));
		splitStr = new String(settings.getProperty("splitString"));
		numNoise = Integer.parseInt(settings.getProperty("numNoise"));
		LEVEL = Integer.parseInt(settings.getProperty("LEVEL"));
		rlow = Double.parseDouble(settings.getProperty("rlow"));
		rhigh = Double.parseDouble(settings.getProperty("rhigh"));
		stepLen = Double.parseDouble(settings.getProperty("stepLen"));
		dataFile = new String(settings.getProperty("ratingData"));
		maxIterTimes = Integer.parseInt(settings.getProperty("maxIterTimes"));
		convergence = Float.parseFloat(settings.getProperty("convergence"));
	}// Of the first constructor

	/**
	 * 
	 * @throws Exception
	 */
	public void readData() throws Exception {
		String tempLine = null;
		String[] tempStr = null;

		// Compute values of arrays
		File tempFile = new File(dataFile);
		if (!tempFile.exists()) {
			System.out.println("File is not exist!");
			return;
		} // Of if

		RandomAccessFile tempRanFile = new RandomAccessFile(tempFile, "r");
		// 读文件的起始位置
		int tempBeginIndex = 0;
		// 将读文件的开始位置移到beginIndex位置。
		tempRanFile.seek(tempBeginIndex);

		// Step 1. read rating data to triple_data
		trData = new Triple[numOfRatings];
		for (int i = 0; i < numOfRatings; i++) {
			trData[i] = new Triple();
		} // Of for i
		uDgr = new int[userNum];
		uRatings = new float[userNum][];
		uRateInds = new int[userNum][];
		
		iDgr = new int[itemNum];
		iRatings = new float[itemNum][];
		iRateInds = new int[itemNum][];
		int tempCount = 0;
		while ((tempLine = tempRanFile.readLine()) != null) {
			tempStr = tempLine.split(splitStr);
			trData[tempCount].user = Integer.parseInt(tempStr[0]) - 1;
			trData[tempCount].item = Integer.parseInt(tempStr[1]) - 1;
			trData[tempCount].rate = Float.parseFloat(tempStr[2]);//*2
			uDgr[trData[tempCount].user]++;
			iDgr[trData[tempCount].item]++;
			tempCount++;
		} // Of while

		for (int i = 0; i < uDgr.length; i++) {
			uRatings[i] = new float[uDgr[i]];
			uRateInds[i] = new int[uDgr[i]];

			uDgr[i] = 0;
		} // Of for i
		
		for (int i = 0; i < iDgr.length; i++) {
			iRatings[i] = new float[iDgr[i]];
			iRateInds[i] = new int[iDgr[i]];

			iDgr[i] = 0;
		} // Of for i

		tempRanFile.close();

		// Step 2. Convert triple_data into user set.
		for (int i = 0; i < numOfRatings; i++) {
			int tempUserIndex = trData[i].user;
			int tempItemIndex = trData[i].item;
			float tempRating = trData[i].rate;
			uRatings[tempUserIndex][uDgr[tempUserIndex]] = tempRating;
			uRateInds[tempUserIndex][uDgr[tempUserIndex]] = tempItemIndex;
			
			iRatings[tempItemIndex][iDgr[tempItemIndex]] = tempRating;
			iRateInds[tempItemIndex][iDgr[tempItemIndex]] = tempUserIndex;

			totalOfRatings += tempRating;
			uDgr[tempUserIndex]++;
			iDgr[tempItemIndex]++;
		} // of for i
	}// of readData

	/**
	 ********************** 
	 * Convert the data set into a vector. The result is stored in dataVector.
	 * 
	 * @see #generateRandomSubMatrix(int)
	 * @see tool.MatrixOpr#getMedian(double[])
	 ********************** 
	 */
	public void computeDataVector() {
		dataVector = new float[numOfRatings];

		int tempCnt = 0;
		for (int i = 0; i < uRatings.length; i++) {
			for (int j = 0; j < uRatings[i].length; j++) {
				dataVector[tempCnt] = uRatings[i][j];
				tempCnt++;
			} // of for j
		} // of for i
	}// of getTrainVector

	/**
	 ********************** 
	 * Compute the average rating of the data set.
	 ********************** 
	 */
	public void computeAverageRating() {
		meanOfRatings = totalOfRatings / numOfRatings;
	}// Of computeAverageRating

	/**
	 ********************** 
	 * Recompute the data set. Each rating subtracts the mean value. In this way the
	 * average value would be 0.
	 ********************** 
	 */
	public void recomputeDataset() {
		for (int i = 0; i < uRatings.length; i++) {
			for (int j = 0; j < uRatings[i].length; j++) {
				uRatings[i][j] = uRatings[i][j] - meanOfRatings;
			} // of for j
		} // of for i

		for (int i = 0; i < dataVector.length; i++) {
			dataVector[i] -= meanOfRatings;
		} // of for i
	}// Of recomputeTrainset

	/**
	 ********************** 
	 * Add the mean rating back so that we can compare the prediction with the
	 * actual one.
	 ********************** 
	 */
	public void recoverRatingMatrix() {
		for (int i = 0; i < uRatings.length; i++) {
			for (int j = 0; j < uRatings[i].length; j++) {
				uRatings[i][j] += meanOfRatings;
			} // of for j
		} // of for i

		for (int i = 0; i < dataVector.length; i++) {
			dataVector[i] += meanOfRatings;
		} // of for i
	}// of recoverTrainMatrix

	/**
	 ********************** 
	 * Generate random sub matrices for initialization. The elements are subject to
	 * the uniform distribution in (-tempMu, tempMu).
	 ********************** 
	 */
	public void generateRandomSubMatrix() {
		// rank = paraRank;
		subU = new float[userNum][rank];
		subV = new float[itemNum][rank];

		double tempMedianOfTrain = tool.MatrixOpr.getMedian(dataVector);
		double tempMu = Math.sqrt(Math.abs(tempMedianOfTrain) / rank);
		System.out.println("generateRandomSubMatrix test 0");
		System.out.println(tempMedianOfTrain + ":" + tempMu);
		// Step 1. Generate two gaussian sub-matrices
		for (int j = 0; j < rank; j++) {
			for (int i = 0; i < userNum; i++) {
				subU[i][j] = (float) (Math.random() * 2 * tempMu - tempMu);
			} // of for i

			for (int i = 0; i < itemNum; i++) {
				subV[i][j] = (float) (Math.random() * 2 * tempMu - tempMu);
			} // of for i
		} // of for j
			// SimpleTool.printMatrix(subU);
	}// of generateRandomSubMatrix

	/**
	 ********************** 
	 * Set the distribution of the noise on each rating For example: 0 0 1 1 0 0 0 1
	 * 0 1 0 0 0 0 1
	 ********************** 
	 */
	public void setRandomNoiseDistribution() {
		Random tempRandom = new Random();
		initialRatingsNoiseDistribution = new float[numOfRatings][numNoise];
		// If numNoise =3, the noise distribution may be [0, 0, 1] for one rating.
		for (int i = 0; i < numOfRatings; i++) {
			initialRatingsNoiseDistribution[i][tempRandom.nextInt(numNoise)] = 1;
		} // of for i
			// SimpleTool.printMatrix(initialRatingsNoiseDistribution);
	}// Of setRandomNoiseDistribution

	/**
	 * 
	 * @return
	 */
	public float[][] getNoiseDistribution() {
		return initialRatingsNoiseDistribution;
	}// Of getNoiseDistribution

	/**
	 ********************** 
	 * Compute the weight of each noise
	 * 
	 * 		0 0 1 
	 * 		1 0 0 
	 * 		0 1 0 
	 * 		1 0 0 
	 * 		0 0 1 
	 * sum 2 1 2 
	 * weight 2/5 1/5 2/5
	 ********************** 
	 */
	public void computeWeight() {
		noiseWeight = new float[numNoise];
		for (int j = 0; j < initialRatingsNoiseDistribution[0].length; j++) {
			float tempColSum = 0;
			// Sum the noise distribution of column
			for (int i = 0; i < initialRatingsNoiseDistribution.length; i++) {
				tempColSum += initialRatingsNoiseDistribution[i][j];
			} // of for i
			noiseWeight[j] = tempColSum / numOfRatings;
		} // of for j
	}// of computeWeight

	/**
	 * 
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		try {
			String tempPropertyFileName = new String("src/properties/ml-10m.properties");
			DataInfo tempData = new DataInfo(tempPropertyFileName);
			tempData.readData();
			tempData.computeAverageRating();
			int[] numofEachLevel = new int[tempData.LEVEL];
			double deviation = 0;

			for (int j = 0; j < tempData.LEVEL; j++) {
				for (int i = 0; i < tempData.numOfRatings; i++) {
					if (Math.abs(tempData.trData[i].rate - (j + 1) * tempData.stepLen) < 0.001) {
						numofEachLevel[j]++;
					} // of if
				} // of for i
				
				System.out.println((j + 1) * tempData.stepLen 
						+ "'s rating number:" + numofEachLevel[j]);
				deviation += numofEachLevel[j] * 
						Math.pow(((j + 1) * tempData.stepLen 
								- tempData.meanOfRatings), 2);
			} // of for j
			
			deviation /= tempData.totalOfRatings;
			System.out.println("average rating:" + tempData.meanOfRatings);
			System.out.println("deviation:" + deviation);
			System.out.println("density:" + (tempData.totalOfRatings + 0.0) / (tempData.userNum * tempData.itemNum));
		} catch (Exception ee) {
			ee.printStackTrace();
		} // Of try
	}// Of main
}// Of Class DataInfo
