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
	public double computeMGofMAE(double[][][] paraVisBasedCondPro, float[] paraTrainVector) {
		double tempMG = 0;

		// SimpleTool.printDoubleArray(paraTrainVector);
		for (int y = 0; y < paraTrainVector.length; y++) {
			for (int z = 0; z < paraVisBasedCondPro[0][(int) paraTrainVector[y] - 1].length; z++) {
				tempMG += paraVisBasedCondPro[0][z][(int) paraTrainVector[y] - 1]
						* Math.abs(z + 1 - paraTrainVector[y]);
			} // of for i
		} // of for l

		tempMG = tempMG / paraTrainVector.length;

		return tempMG;
	}// of computeMGofMAE

	/**
	 * Compute magic barrier of MAE Based on \gamma_{ijk}
	 * 
	 */
	public double computeMGofMAE2(CondProbSingleSigma paraCP, double[][] paraVisBasedCondPro,
			float[] paraTrainVector) {
		double tempMG = 0;

		// SimpleTool.printDoubleArray(paraTrainVector);
		for (int y = 0; y < paraTrainVector.length; y++) {// ÆÀ·ÖÏòÁ¿µÄ³¤¶È
			for (double z = paraCP.data.rlow; z <= paraCP.data.rhigh; z = z + paraCP.data.stepLen) {
				int tempY = paraCP.getIndexBasedRating(paraTrainVector[y]);//visual rating
				int tempZ = paraCP.getIndexBasedRating(z);//true rating

				tempMG += paraVisBasedCondPro[tempZ][tempY] * Math.abs(z - paraTrainVector[y]);
			} // of for i
		} // of for l

		tempMG = tempMG / paraTrainVector.length;

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
	public double computeMGofRMSE2(CondProbSingleSigma paraCP, double[][] paraVisBasedCondPro,
			float[] paraTrainVector) {
		double tempMG = 0;

		// SimpleTool.printDoubleArray(paraTrainVector);
		for (int y = 0; y < paraTrainVector.length; y++) {
			for (double z = paraCP.data.rlow; z <= paraCP.data.rhigh; z = z + paraCP.data.stepLen) {
				int tempY = paraCP.getIndexBasedRating(paraTrainVector[y]);//visual rating
				int tempZ = paraCP.getIndexBasedRating(z);//true rating

				tempMG += paraVisBasedCondPro[tempZ][tempY] * Math.pow(z - paraTrainVector[y], 2);
			} // of for i
		} // of for l

		tempMG = Math.sqrt(tempMG / paraTrainVector.length);

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
	 * ï¿½ï¿½Ð¡ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½
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
			String tempPropertyFileName = new String("src/properties/ml-1m.properties");
			DataInfo tempData = new DataInfo(tempPropertyFileName);
			tempData.readData();
			tempData.computeDataVector();

			// Compute magic barrier
			CondProbSingleSigma tempCon = new CondProbSingleSigma(tempData);
			float tempSigma = (float)(0.9 * 0.9);
			//step 1. Calculate the conditional probability P_{¦Ò_k}(r_{i,j}= y|o_{i,j} = z)
			tempCon.computeTrueBasedCondProWithInteg(tempSigma);
			System.out.println("P_{¦Ò_k}(r_{i,j}= y|o_{i,j} = z) is: ");
			SimpleTool.printMatrix(tempCon.trueBasedCondPro);
			//step 2. Calculate the probability P_{¦Ò_k}(r_{i,j} = y)
			tempCon.computePropOfVisRating();
			System.out.println("P_{¦Ò_k}(r_{i,j} = y is: ");
			SimpleTool.printDoubleArray(tempCon.propOfEachVisRating);
			//step 3. Calculate the probability P_{¦Ò_k}(o_{i,j} = z)
			tempCon.computePropOfTrueRatingSOR();
			System.out.println("P_{¦Ò_k}(o_{i,j} = z) is: ");
			SimpleTool.printDoubleArray(tempCon.propOfTrueRating);
			//step 4. calculate the probability P_{¦Ò_k}(o_{i,j}= z|r_{i,j} = y)
			tempCon.computeVisBasedCondProBasedOnBayes();
			System.out.println("P_{¦Ò_k}(o_{i,j}= z|r_{i,j} = y) is: ");
			SimpleTool.printMatrix(tempCon.visBasedCondPro);
			
			MBSingleSigmaCompution tempMBCom = new MBSingleSigmaCompution();
			double tempMBOfMAE = tempMBCom.computeMGofMAE2(tempCon, tempCon.visBasedCondPro, tempData.dataVector);
			System.out.println("MGBR(based on gamma_{ijk}) of MAE: " + tempMBOfMAE);
			double tempMBOfRMSE = tempMBCom.computeMGofRMSE2(tempCon, tempCon.visBasedCondPro, tempData.dataVector);
			System.out.println("MGBR(based on gamma_{ijk}) of RMSE: " + tempMBOfRMSE);
		} catch (Exception e) {
			e.printStackTrace();
		} // of try

	}// of main

}// Of class MBCompution
