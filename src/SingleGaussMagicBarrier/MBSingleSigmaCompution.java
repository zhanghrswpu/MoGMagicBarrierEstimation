package SingleGaussMagicBarrier;

import java.io.IOException;

import algorithm.MLGMDN;
import datamodel.DataInfo;
import tool.SimpleTool;
import MoGMagicBarrier.*;

public class MBSingleSigmaCompution {
	/**
	 * Compute magic barrier Based on z_{ijk}
	 * 
	 */
	public double computeMG(double[][][] paraVisBasedCondPro, int[][] paraZ, double[] paraTrainVector) {
		double tempMG = 0;

		// SimpleTool.printDoubleArray(paraTrainVector);
		for (int l = 0; l < paraZ.length; l++) {
			for (int k = 0; k < paraZ[0].length; k++) {
				if (paraZ[l][k] == 1) {
					for (int i = 0; i < paraVisBasedCondPro[k][(int) paraTrainVector[l] - 1].length; i++) {
						tempMG += paraVisBasedCondPro[k][(int) paraTrainVector[l] - 1][i]
								* Math.abs(i - paraTrainVector[l]);
					} // of for i
				} // of if
			} // of for k
		} // of for l

		tempMG = tempMG / paraZ.length;

		return tempMG;
	}// of computeMG

	/**
	 * Compute magic barrier of MAE Based on \gamma_{ijk}
	 * 
	 */
	public double computeMGofMAE(double[][][] paraVisBasedCondPro, float[][] paraNoiseDistribution,
			float[] paraTrainVector) {
		double tempMG = 0;

		// SimpleTool.printDoubleArray(paraTrainVector);
		for (int y = 0; y < paraNoiseDistribution.length; y++) {
			for (int k = 0; k < paraNoiseDistribution[0].length; k++) {
				for (int z = 0; z < paraVisBasedCondPro[k][(int) paraTrainVector[y] - 1].length; z++) {
					tempMG += paraNoiseDistribution[y][k] * paraVisBasedCondPro[k][z][(int) paraTrainVector[y] - 1]
							* Math.abs(z + 1 - paraTrainVector[y]);
				} // of for i
			} // of for k
		} // of for l

		tempMG = tempMG / paraNoiseDistribution.length;

		return tempMG;
	}// of computeMGofMAE

	/**
	 * Compute magic barrier of MAE Based on \gamma_{ijk}
	 * 
	 */
	public double computeMGofMAE2(CondProb paraCP, double[][][] paraVisBasedCondPro, float[][] paraNoiseDistribution,
			float[] paraTrainVector) {
		double tempMG = 0;

		// SimpleTool.printDoubleArray(paraTrainVector);
		for (int y = 0; y < paraNoiseDistribution.length; y++) {// 评分向量的长度
			for (int k = 0; k < paraNoiseDistribution[0].length; k++) {// 噪声的个数
				for (double z = paraCP.data.rlow; z <= paraCP.data.rhigh; z = z + paraCP.data.stepLen) {
					int tempY = paraCP.getIndexBasedRating(paraTrainVector[y]);
					int tempZ = paraCP.getIndexBasedRating(z);

					tempMG += paraNoiseDistribution[y][k] * paraVisBasedCondPro[k][tempZ][tempY]
							* Math.abs(z - paraTrainVector[y]);
				} // of for i
			} // of for k
		} // of for l

		tempMG = tempMG / paraNoiseDistribution.length;

		return tempMG;
	}// of computeMGofMAE2

	/**
	 * Compute magic barrier of RMSE Based on \gamma_{ijk}
	 * 
	 */
	public double computeMGofRMSE(double[][][] paraVisBasedCondPro, float[][] paraNoiseDistribution,
			float[] paraTrainVector) {
		double tempMG = 0;

		// SimpleTool.printDoubleArray(paraTrainVector);
		for (int y = 0; y < paraNoiseDistribution.length; y++) {
			for (int k = 0; k < paraNoiseDistribution[0].length; k++) {
				for (int z = 0; z < paraVisBasedCondPro[k][(int) paraTrainVector[y] - 1].length; z++) {
					tempMG += paraNoiseDistribution[y][k] * paraVisBasedCondPro[k][z][(int) paraTrainVector[y] - 1]
							* Math.pow(z + 1 - paraTrainVector[y], 2);
				} // of for i
			} // of for k
		} // of for l

		tempMG = Math.sqrt(tempMG / paraNoiseDistribution.length);

		return tempMG;
	}// of computeMGofRMSE

	/**
	 * Compute magic barrier of RMSE Based on \gamma_{ijk}
	 * 
	 */
	public double computeMGofRMSE2(CondProb paraCP, double[][][] paraVisBasedCondPro, float[][] paraNoiseDistribution,
			float[] paraTrainVector) {
		double tempMG = 0;

		// SimpleTool.printDoubleArray(paraTrainVector);
		for (int y = 0; y < paraNoiseDistribution.length; y++) {
			for (int k = 0; k < paraNoiseDistribution[0].length; k++) {
				for (double z = paraCP.data.rlow; z <= paraCP.data.rhigh; z = z + paraCP.data.stepLen) {
					int tempY = paraCP.getIndexBasedRating(paraTrainVector[y]);
					int tempZ = paraCP.getIndexBasedRating(z);

					tempMG += paraNoiseDistribution[y][k] * paraVisBasedCondPro[k][tempZ][tempY]
							* Math.pow(z - paraTrainVector[y], 2);
				} // of for i
			} // of for k
		} // of for l

		tempMG = Math.sqrt(tempMG / paraNoiseDistribution.length);

		return tempMG;
	}// of computeMGofRMSE

	/**
	 * Compute magic barrier Based on ||O - R||
	 * 
	 */
	public double computeMG(double[][] paraVisRatingMatrix, double[][] paraTrueRatingMatrix) {
		double tempMG = 0;
		int tempCount = 0;

		for (int i = 0; i < paraVisRatingMatrix.length; i++) {
			for (int j = 0; j < paraVisRatingMatrix[0].length; j++) {
				if (paraVisRatingMatrix[i][j] > 1e-6) {
					tempMG += Math.abs(paraVisRatingMatrix[i][j] - paraTrueRatingMatrix[i][j]);
					tempCount++;
				} // of if
			} // of for j
		} // of for i

		return tempMG / tempCount;
	}// of computeMG

	/**
	 *************************** 
	 * Bubble sort. <br>
	 * 锟斤拷小锟斤拷锟斤拷锟斤拷锟斤拷
	 * 
	 * @param paraSigma The given array.
	 * @author Zhanghr 2014/06/14
	 * @return The constructed array.
	 *************************** 
	 */
	public static void bubbleSort(float[] paraSigma, float[] paraPi) {
		int tempLength = paraSigma.length;
		float tempSigma = 0;
		float tempPi = 0;
		for (int i = 0; i <= tempLength - 1; i++) {
			for (int j = tempLength - 1; j > i; j--) {
				if (paraSigma[j] < paraSigma[j - 1]) {
					tempSigma = paraSigma[j];
					paraSigma[j] = paraSigma[j - 1];
					paraSigma[j - 1] = tempSigma;

					tempPi = paraPi[j];
					paraPi[j] = paraPi[j - 1];
					paraPi[j - 1] = tempPi;
				} // of if
			} // of for j
		} // of for i
	}// of bubbleSort

	/**
	 ************************** 
	 * @param args
	 * @throws IOException
	 ************************** 
	 */
	public static void main(String[] args) throws IOException {
		try {

			// Prepare data and preprocessing
			String tempPropertyFileName = new String("src/properties/ml-100k.properties");
			DataInfo tempData = new DataInfo(tempPropertyFileName);
			tempData.readData();
			
			// SimpleTool.printMatrix(tempData.uTrRatings);
			tempData.computeAverageRating();
			tempData.computeDataVector();
			tempData.recomputeDataset();// subtract the average of training set
			tempData.generateRandomSubMatrix();
			// SimpleTool.printMatrix(tempData.subU);
			// SimpleTool.printMatrix(tempData.subV);
			tempData.numNoise = 1;
			tempData.setRandomNoiseDistribution();
			tempData.computeWeight();// For the first time. Round 0

			MLGMDN tempEm = new MLGMDN(tempData);
			// tempEm.setModel();

			tempEm.iterationEM(1);
//			tempEm.printNoiseDistribution();
			tempData.recoverRatingMatrix();
//			System.out.println("Latent variable z_{ijk}:");
//			SimpleTool.printMatrix(tempEm.z);

			// Compute magic barrier
			CondProb tempCP = new CondProb(tempData);
			float[] singleSigma = { (float) (0.9 * 0.9) };
			double[][][] tempTrueBasedCondPro = tempCP.computeTrueBasedCondProWithInteg(singleSigma);
			// SimpleTool.printFloatArray(tempCP.sigma);//correct
			// SimpleTool.printFloatArray(tempEm.noiseWeight);//correct
			System.out.println("TrueBasedCondPro start ...");
			SimpleTool.printTripleMatrix(tempTrueBasedCondPro);// correct
			System.out.println("TrueBasedCondPro end ...");
			tempCP.computePropOfVisRating(tempEm);// correct
			System.out.println("propOfEachVisRating start ...");
			SimpleTool.printMatrix(tempCP.propOfEachVisRating);// correct
			System.out.println("propOfEachVisRating end ...");
			tempCP.computePropOfTrueRatingSOR();// correct
			System.out.println("propOfTrueRating start ...");
			SimpleTool.printMatrix(tempCP.propOfTrueRating);// correct
			System.out.println("propOfTrueRating end ...");
			tempCP.computeVisBasedCondProBasedOnBayes();
			System.out.println("visBasedCondPro start ...");
			SimpleTool.printTripleMatrix(tempCP.visBasedCondPro);
			System.out.println("visBasedCondPro end ...");
			MBSingleSigmaCompution tempMBCom = new MBSingleSigmaCompution();
//			double tempMB = tempMBCom.computeMG(tempCP.visBasedCondPro, tempEm.z, tempData.trainVector);
//			System.out.println("MB(based on z_{ijk}): " + tempMB); 

			double tempMB = tempMBCom.computeMGofMAE2(tempCP, tempCP.visBasedCondPro, tempEm.noiseDistribution,
					tempData.dataVector);
			System.out.println("MB(based on gamma_{ijk}) of MAE: " + tempMB);
			tempMB = tempMBCom.computeMGofRMSE2(tempCP, tempCP.visBasedCondPro, tempEm.noiseDistribution,
					tempData.dataVector);
			System.out.println("MB(based on gamma_{ijk}) of RMSE: " + tempMB);

			bubbleSort(tempCP.sigma, tempEm.noiseWeight);
			SimpleTool.printFloatArray(tempCP.sigma);// correct
			SimpleTool.printFloatArray(tempEm.noiseWeight);// correct
			// tempMB = tempMBCom.computeMG(tempData.ratingMatrix, tempEm.predictions);
			// System.out.println("MB(||O-R||): " + tempMB);
			// System.out.println("当前噪声数为" + j + "的第" + i + "次" +"\r\n");

		} catch (Exception e) {
			e.printStackTrace();
		} // of try

	}// of main

}// Of class MBCompution
