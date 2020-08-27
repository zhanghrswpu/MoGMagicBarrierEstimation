package SingleGaussMagicBarrier;

import MoGMagicBarrier.Iteration;
import MoGMagicBarrier.NormalDistribution;
import datamodel.DataInfo;

/**
 * @author Heng-Ru Zhang 2016/12/19. 整个的理论分析见：《张恒汝——研究报告(118) - magic
 *         barrier理论分析及实验研究（重新整理）》
 */

public class LevelMagicBarrier extends CondProbSingleSigma {
	/**
	 * item-based total and average ratings
	 */
	private double[] iTotRates;
	private double[] iAveRates;

	/**
	 * Conditional probability based “true” rating
	 */
	double[][][] trueBasedCondProForSigmas;

	/**
	 * 5 * 5 matrix for rating number of each level , and array for total number of
	 * each level
	 */
	private int[][] ratingNumberOfLevels;
	private int[] totalNumberOfLevels;
	private double[][] levelBasedCondPro;

	/**
	 ************************* 
	 * Construct the rating matrix.
	 * 
	 * @param paraRatingFilename the rating filename.
	 ************************* 
	 */
	public LevelMagicBarrier(DataInfo paraData) throws Exception {
		super(paraData);
		buildLevelModel();
	}// Of the first constructor

	/**
	 * Build level model
	 */
	private void buildLevelModel() {

		// Step 0. Compute the average rating of each user
		iTotRates = new double[data.itemNum];
		iAveRates = new double[data.itemNum];
		for (int i = 0; i < data.iRatings.length; i++) {
			for (int j = 0; j < data.iRatings[i].length; j++) {
				// System.out.println(i + ":" + j + "=" +
				// userRatingInformation[i][j]);
				iTotRates[i] += data.iRatings[i][j];
			} // Of for j
			if (data.iDgr[i] > 1e-6) {
				iAveRates[i] = iTotRates[i] / data.iDgr[i];
			} // Of if
		} // Of for i
	}

	/**
	 * Compute the rating number and total number of levels
	 */
	public void computeRatingNumberLevels() {
		ratingNumberOfLevels = new int[data.LEVEL][data.LEVEL];
		totalNumberOfLevels = new int[data.LEVEL];
		int[] tempCount;

		for (int y = 0; y < iAveRates.length; y++) {
			for(int z = 0; z < data.LEVEL; z ++){
				if(iAveRates[y] >= (data.rlow + data.stepLen * (z - 0.5))
						&& iAveRates[y] < (data.rlow + data.stepLen * (z + 0.5))) {
					totalNumberOfLevels[z] += data.iDgr[y];
					tempCount = countRatingNumber(data.iRatings[y]);
					for(int k = 0; k < data.LEVEL; k ++) {
						ratingNumberOfLevels[z][k] += tempCount[k];
					}//of for k
				}//of if	
			}//of for z
		} // Of for y

		// Check
//		int tempTotal = 0;
//		for(int i = 0; i < CHANNEL; i ++){
//			for(int j = 0; j < CHANNEL; j ++){
//				tempTotal += ratingNumberOfLevels[i][j];
//			}//Of for j
//		}//Of for i
//		System.out.println("The total rating is: " + tempTotal);
	}// of computeRatingNumberLevels

	/**
	 * Count the rating number of users for each item
	 */
	private int[] countRatingNumber(float[] paraRatings) {
		int[] tempCounts = new int[data.LEVEL];
		for (int i = 0; i < paraRatings.length; i++) {
			for (int j = 0; j < data.LEVEL; j++) {
				if(paraRatings[i] > (data.rlow + data.stepLen * (j - 0.5))
						&& paraRatings[i] <= (data.rlow + data.stepLen * (j + 0.5))){
					tempCounts[j]++;
				} // Of if
			} // Of for j
		} // Of for i

		return tempCounts;
	}// Of countRatingNumberLevels

	/**
	 * Compute level-based conditional probability
	 */
	public void computeLevelBasedCondPro() {
		levelBasedCondPro = new double[data.LEVEL][data.LEVEL];
		for (int i = 0; i < data.LEVEL; i++) {
			for (int j = 0; j < data.LEVEL; j++) {
				levelBasedCondPro[i][j] = (ratingNumberOfLevels[i][j] + 0.0) / totalNumberOfLevels[i];
			} // Of for j
		} // Of for i
//		System.out.println("The level-based conditional probability is: ");
//		printMatrix(levelBasedCondPro);
	}// Of computeLevelBasedCondPro

	/**
	 * Compute the true-based conditional probabilities for all sigma settings
	 */
	public void computeTrueBasedCondProForSigmas(int paraNumberOfSigmas) {
		trueBasedCondProForSigmas = new double[paraNumberOfSigmas][data.LEVEL][data.LEVEL];
		double tempSigma = 0;
		for (int i = 0; i < paraNumberOfSigmas; i++) {
			tempSigma = Math.pow(0.1 + 0.1 * i, 2);
			computeTrueBasedCondProWithInteg(tempSigma);
			for (int j = 0; j < data.LEVEL; j++) {
				for (int k = 0; k < data.LEVEL; k++) {
					trueBasedCondProForSigmas[i][j][k] = trueBasedCondPro[j][k];
				} // Of for k
			} // Of for j

			System.out.println("trueBasedCondProForSigmas is as followed:");
			printMatrix(trueBasedCondProForSigmas[i]);
		} // Of for i
	}// of computeTrueBasedCondProForSigmas

	/**
	 * Eucliden distance
	 */
	public double[] findOptimalSigmas() {
		double[] tempOptimalSigmas = new double[data.LEVEL];
		double tempDistance = 0;
		double tempMinDistance = 999;
		int[] tempMinIndex = new int[data.LEVEL];
		for (int i = 0; i < data.LEVEL; i++) {
			// Step 1. Compute the Euclidean distances
			tempMinDistance = 999;
//			System.out.println("Level-" + (i + 1));
			for (int j = 0; j < trueBasedCondProForSigmas.length; j++) {
				tempDistance = euclideanDistance(trueBasedCondProForSigmas[j][i], levelBasedCondPro[i]);
//				System.out.print((j) + "-" + tempDistance + " ");
				if (tempMinDistance > tempDistance) {
					tempMinIndex[i] = j;
					tempMinDistance = tempDistance;
				} // Of if
			} // Of for j

			// Step 2. Compute the optimal sigmas
			tempOptimalSigmas[i] = (1 + tempMinIndex[i] + 0.0) / 10;
			System.out.println("level-" + (i + 1) + "  sigma_k: " + tempOptimalSigmas[i]);

			// Step 3. The optimal level-based conditional probability
			for (int j = 0; j < data.LEVEL; j++) {
				levelBasedCondPro[i][j] = trueBasedCondProForSigmas[tempMinIndex[i]][i][j];
			} // Of for j
		} // Of for i

		System.out.println("The optimal level-based conditional probablity: ");
		printMatrix(levelBasedCondPro);

		return tempOptimalSigmas;
	}// Of findOptimalSigmas

	/**
	 * Euclidean distance
	 */
	private double euclideanDistance(double[] paraArray1, double[] paraArray2) {
		double tempDistance = 0;

		for (int i = 0; i < paraArray1.length; i++) {
			tempDistance += Math.abs(paraArray1[i] - paraArray2[i]);
		} // of for i
		return tempDistance;
	}// Of euclidenDistance

	/**
	 ************************* 
	 * Print a double array.
	 ************************* 
	 */
	public static void printDoubleArray(double[] paraDoubleArray) {
		for (int i = 0; i < paraDoubleArray.length; i++) {
			System.out.print(paraDoubleArray[i] + " ");
		} // Of for i
		System.out.println();
	}// Of printAllReducts

	/**
	 *************************** 
	 * Print an int matrix, simply for test.
	 * 
	 * @param paraMatrix The given matrix. Different rows may contain different
	 *                   number of values.
	 *************************** 
	 */
	public static void printMatrix(double[][] paraMatrix) {
		if (paraMatrix.length == 0) {
			System.out.println("This is an empty matrix.");
			return;
		} else {
			System.out.println("This is an int matrix: ");
		} // Of if

		for (int i = 0; i < paraMatrix.length; i++) {
			for (int j = 0; j < paraMatrix[i].length; j++) {
				System.out.print("" + paraMatrix[i][j] + "\t");
			} // Of for j
			System.out.println();
		} // Of for i
	}// Of printMatrix

	/**
	 * Gauss-seidel iteration
	 */
	public void computePropOfTrueRatingGaussSeidel() {
		propOfTrueRating = new double[data.LEVEL];
		for (int i = 0; i < data.LEVEL; i++) {
			propOfTrueRating[i] = 0.2;
		} // Of if

		double[][] levelBasedCondPro2 = transposition(levelBasedCondPro);
		propOfTrueRating = Iteration.Gauss_seidel(levelBasedCondPro2, propOfTrueRating, propOfEachVisRating, 1e-5);

		// Step 4. Check
		double[] tempPropOfVisibleRating = new double[data.LEVEL];
		for (int i = 0; i < data.LEVEL; i++) {
			for (int j = 0; j < data.LEVEL; j++) {
				tempPropOfVisibleRating[i] += propOfTrueRating[j] * levelBasedCondPro[j][i];
			} // Of for j
		} // Of for i
			// System.out.println("The proportions of visual rating (check): ");
			// printDoubleArray(tempPropOfVisibleRating);

		// Step 5. If propOfTrueRating < 0, then propOfTrueRating = 0
		for (int i = 0; i < data.LEVEL; i++) {
			if (propOfTrueRating[i] < 1e-6) {
				propOfTrueRating[i] = 0;
			} // Of if
		} // Of for i

		// Step 6. normalization
		double tempTotalProp = 0;
		for (int i = 0; i < data.LEVEL; i++) {
			tempTotalProp += propOfTrueRating[i];
		} // of for i
		for (int i = 0; i < data.LEVEL; i++) {
			propOfTrueRating[i] /= tempTotalProp;
			propOfTrueRating[i] = Math.round(propOfTrueRating[i] * 10000) / 10000.0;
		} // of for i

		// System.out
		// .println("The proportions of true rating (afer normalization): ");
		// printDoubleArray(propOfTrueRating);

		// Step 7. Check 2
		for (int i = 0; i < data.LEVEL; i++) {
			tempPropOfVisibleRating[i] = 0;
			for (int j = 0; j < data.LEVEL; j++) {
				tempPropOfVisibleRating[i] += propOfTrueRating[j] * levelBasedCondPro[j][i];
			} // Of for j
		} // Of for i
			// System.out.println("The proportions of visual rating (check2): ");
			// printDoubleArray(tempPropOfVisibleRating);
	}// Of computePropOfTrueRatingGaussSeidel

	/**
	 * Employ Bayes method to compute visible-rating based conditional probability
	 * 
	 */
	public void computeVisBasedCondProBasedOnBayes() {
		visBasedCondPro = new double[data.LEVEL][data.LEVEL];

		// Step 1. Employ Bayes method to compute conditional probability based
		// on real rating
		for (int i = 0; i < visBasedCondPro.length; i++) {
			for (int j = 0; j < visBasedCondPro[0].length; j++) {
				visBasedCondPro[i][j] = propOfTrueRating[j] * levelBasedCondPro[j][i] / propOfEachVisRating[i];
			} // of for j
		} // Of for i

		System.out.println("visBasedCondPro before normalization: ");
		printMatrix(visBasedCondPro);

		// ???归一化
		double tempProbability = 0;
		for (int i = 0; i < data.LEVEL; i++) {
			tempProbability = 0;
			for (int j = 0; j < data.LEVEL; j++) {
				tempProbability += visBasedCondPro[i][j];
			} // Of for j
			for (int j = 0; j < data.LEVEL; j++) {
				visBasedCondPro[i][j] /= tempProbability;
				visBasedCondPro[i][j] = Math.round(visBasedCondPro[i][j] * 10000) / 10000.0;
			} // Of for j
		} // Of for i

		System.out.println("visBasedCondPro after normalization: ");
		printMatrix(visBasedCondPro);
	}// Of computeRealBasedCondProbabilityBasedOnBayes
	
	/**
	 * Compute MAE
	 * 
	 */
	public double computeMG() {
		double tempMG = 0;
		for (int i = 0; i < visBasedCondPro.length; i++) {
			for (int j = 0; j < visBasedCondPro[0].length; j++) {
				tempMG += visBasedCondPro[i][j] * gammaOfEachVisRating[i]
						* Math.abs(j - i);
			}// of for j
		}// of for i

		tempMG = tempMG / totalGammaOfVisRating;

		return tempMG;
	}// of computeMAE

	public static void main(String[] args) {
		try {
			// TODO 自动生成的方法存根
			// TODO 自动生成的方法存根
			String tempPropertyFileName = new String("src/properties/ml-100k.properties");
			DataInfo tempData = new DataInfo(tempPropertyFileName);
			tempData.readData();
			tempData.computeAverageRating();
			
			LevelMagicBarrier tempMagic = new LevelMagicBarrier(tempData);
			
			tempMagic.computeRatingNumberLevels();
			tempMagic.computeLevelBasedCondPro();
			tempMagic.computeTrueBasedCondProForSigmas(9);
			tempMagic.findOptimalSigmas();
			tempMagic.computePropOfVisRating();
			tempMagic.computePropOfTrueRatingGaussSeidel();
			tempMagic.computeVisBasedCondProBasedOnBayes();
			double tempMG = tempMagic.computeMG();
			System.out.println("MG: " + tempMG);
		} catch (Exception e) {
			e.printStackTrace();
		} // Of try
	}// Of main
}// Of class MagicOfUniqueMovieLens943u1682m