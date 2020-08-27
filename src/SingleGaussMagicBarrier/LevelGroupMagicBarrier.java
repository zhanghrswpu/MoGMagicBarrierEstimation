package SingleGaussMagicBarrier;

import MoGMagicBarrier.Iteration;
import datamodel.DataInfo;

/**
 * @author Heng-Ru Zhang 2016/12/19.
 * 整个的理论分析见：《张恒汝——研究报告(118) - magic barrier理论分析及实验研究（重新整理）》
 */


public class LevelGroupMagicBarrier extends CondProbSingleSigma {
	/**
	 * Item total and average ratings for group models
	 */
	private double[][] itemGroupTotRatings;
	private int[][] itemGroupDegree; // group id, item id
	private double[][] itemGroupAveRatings;

	/**
	 * User-based total and average ratings
	 */
	private double[] userTotRatings;
	private double[] userAveRatings;
	private int[] userGroupIndices;
	private int numOfGroups;

	/**
	 * Conditional probability based “true” rating
	 */
	double[][][] trueBasedPros;

	/**
	 * Group-based conditional probability
	 */
	double[][][] trueGroupBasedCountPro;
	int[][] trueGroupLevelDegree; // level ratings of each group
	int[][][] trueGroupLevelLevelDegree; // level-level ratings of each group

	/**
	 * To count rating distribution, we need to obtain the rating number of each
	 * scale.
	 */
	double[][] visPropGroup;
	double[][] truePropGroup;

	/**
	 * Conditional probability based "visible" rating for group-model
	 */
	double[][][] visGroupBasedBayesPro;
	int[][][] itemGroupLevelDegree;
	int visTotalDegree; // total ratings
	int[] visGroupDegree; // group ratings
	int[][] visGroupLevelDegree; // group-level ratings

	/**
	 ************************* 
	 * Construct the rating matrix.
	 * 
	 * @param paraRatingFilename
	 *            the rating filename.
	 ************************* 
	 */
	public LevelGroupMagicBarrier(DataInfo paraData, int paraNumOfGroups) throws Exception {

		super(paraData);
		/* Initialize */
		numOfGroups = paraNumOfGroups;
		buildGroupModel();
		computeItemAveRatingsForGroup();
		computeTrueBasedProForSigmas(9);
	}// Of the first constructor

	/**
	 * Build group model
	 */
	private void buildGroupModel() {
		userGroupIndices = new int[data.userNum];

		// Step 0. Compute the average rating of each user
		userTotRatings = new double[data.uRatings.length];
		userAveRatings = new double[data.uRatings.length];
		for (int i = 0; i < data.uRatings.length; i++) {
			for (int j = 0; j < data.uRatings[i].length; j++) {
				// System.out.println(i + ":" + j + "=" +
				// userRatingInformation[i][j]);
				userTotRatings[i] += data.uRatings[i][j];
				// Count the number of all ratings
				visTotalDegree++;
			}// Of for j
			if (data.uDgr[i] > 1e-6) {
				userAveRatings[i] = userTotRatings[i] / data.uDgr[i];
			}// Of if
		}// Of for i

		// Step 1. Compute the step length
		double tempLen = (data.rhigh - data.rlow) / numOfGroups;

		// Step 2. Divide the users into different groups
		for (int i = 0; i < data.userNum; i++) {
			for (int j = 0; j < numOfGroups; j++) {
				if (userAveRatings[i] >= (data.rlow + tempLen * j)
						&& userAveRatings[i] < (data.rlow + tempLen * (j + 1))) {
					userGroupIndices[i] = j;
				}// of if
			}// Of for j
			if (userAveRatings[i] < data.rlow) {
				userGroupIndices[i] = 0;
			}// of if
			if (userAveRatings[i] == data.rhigh) {
				userGroupIndices[i] = numOfGroups - 1;
			}// Of if
		}// Of for i
			// Check
			// printArray(userAveRatings);
		// System.out.println("The user group indices: ");
		// printArray(userGroupIndices);
	}// of buildGroupModel

	/**
	 * Compute item average ratings for group-model
	 */
	private void computeItemAveRatingsForGroup() {
		itemGroupTotRatings = new double[numOfGroups][data.itemNum];
		itemGroupDegree = new int[numOfGroups][data.itemNum];
		itemGroupAveRatings = new double[numOfGroups][data.itemNum];
		itemGroupLevelDegree = new int[numOfGroups][data.itemNum][data.LEVEL];

		for (int i = 0; i < data.itemNum; i++) {
			// Step 1. Compute the total ratings and degrees for group model
			for (int j = 0; j < data.iDgr[i]; j++) {
				for (int k = 0; k < numOfGroups; k++) {
					if (userGroupIndices[data.iRateInds[i][j]] == k) {
						itemGroupTotRatings[k][i] += data.iRatings[i][j];
						// For example, group-0, item-0
						itemGroupDegree[k][i]++;
						// Step 1.1 Count the item-group-level degree for
						// computing the group conditional probability
						for (int l = 0; l < data.LEVEL; l++) {
							if (data.iRatings[i][j] == (l + 1)) {
								// For example, group-0, item-0, level-1
								itemGroupLevelDegree[k][i][l]++;
							}// Of if
						}// Of for l
					}// Of if
				}// Of for k
					// Step 2. Compute the average ratings for group model
				for (int k = 0; k < numOfGroups; k++) {
					if (itemGroupDegree[k][i] > 1e-6) {
						itemGroupAveRatings[k][i] = itemGroupTotRatings[k][i]
								/ itemGroupDegree[k][i];
					}// Of if
				}// Of for k
			}// Of for j
		}// Of for i

		// Check
		// printMatrix(itemGroupAveRatings);
		int tempTotal = 0;
		for (int i = 0; i < itemGroupLevelDegree.length; i++) {
			for (int j = 0; j < itemGroupLevelDegree[i].length; j++) {
				for (int k = 0; k < itemGroupLevelDegree[i][j].length; k++) {
					tempTotal += itemGroupLevelDegree[i][j][k];
				}// Of for k
			}// Of for j
		}// Of for i
			// System.out.println("Total ratings: " + tempTotal);
	}// of computeItemAveRatingsForGroup

	/**
	 * Compute the group-based conditional probability
	 */
	public void computeTrueGroupCountPro() {
		trueGroupBasedCountPro = new double[numOfGroups][data.LEVEL][data.LEVEL];
		// The degree of level-group model
		// (For example: itemGroupAveRatings = = 1)
		trueGroupLevelDegree = new int[numOfGroups][data.LEVEL];

		// Step 1. Compute the step length
		double[] tempLen = new double[data.LEVEL + 1];
		for (int i = 0; i < tempLen.length; i++) {
			if (i == 0 || i == (tempLen.length - 1)) {
				tempLen[i] = i + 1;
			} else {
				tempLen[i] = i + 0.5;
			}// Of if
		}// Of for i
			// {1.0, 1.5, 2.5, 3.5, 4.5, 5.0};

		// The degree of level-level-group model
		// (For example: itemGroupAveRatings == 1 and itemRatingInformation ==
		// 1)
		trueGroupLevelLevelDegree = new int[numOfGroups][data.LEVEL][data.LEVEL];
		for (int i = 0; i < numOfGroups; i++) {
			// Step 2. Count the total number and level-based number of ratings
			// when "itemGroupAveRatings" is same
			for (int j = 0; j < data.itemNum; j++) {
				for (int k = 0; k < data.LEVEL; k++) {
					if (itemGroupAveRatings[i][j] >= tempLen[k]
							&& itemGroupAveRatings[i][j] < tempLen[k + 1]) {
						// &&&level ratings for each group
						trueGroupLevelDegree[i][k] += itemGroupDegree[i][j];
						for (int l = 0; l < data.LEVEL; l++) {
							// &&&level-level ratings for each group
							trueGroupLevelLevelDegree[i][k][l] += itemGroupLevelDegree[i][j][l];
						}// of for l
					}// Of if
				}// Of for k
				if (itemGroupAveRatings[i][j] == tempLen[data.LEVEL]) {
					// System.out.println("The group-" + i +" item-" + j +
					// " ratings: " + itemGroupDegree[i][j]);
					trueGroupLevelDegree[i][data.LEVEL - 1] += itemGroupDegree[i][j];
					for (int l = 0; l < data.LEVEL; l++) {
						// &&&level-level ratings for each group
						trueGroupLevelLevelDegree[i][data.LEVEL - 1][l] += itemGroupLevelDegree[i][j][l];
					}// of for l
				}// Of if
			}// Of for j

			// Step 3. Compute the group-based conditional probability
			for (int j = 0; j < data.LEVEL; j++) {
				for (int k = 0; k < data.LEVEL; k++) {
					if (trueGroupLevelDegree[i][j] > 1e-6) {
						trueGroupBasedCountPro[i][j][k] = trueGroupLevelLevelDegree[i][j][k]
								* 1.0 / trueGroupLevelDegree[i][j];
					}// Of if
				}// Of for k
			}// Of for j
				// Check
				// System.out.println("The " + i +
				// " group's conditional probability:");
			// printMatrix(trueGroupBasedCountPro[i]);
		}// of for i
			// printMatrix(visGroupLevelDegree);
	}// Of ComputeGroupCondPro

	/**
	 * Compute true-rating-based condition probability with step-length
	 * integration
	 */
	private void computeTrueBasedProForSigmas(int paraNumberOfSigmas) {
		trueBasedPros = new double[paraNumberOfSigmas][data.LEVEL][data.LEVEL];
		double tempSigma = 0;
		for (int i = 0; i < paraNumberOfSigmas; i++) {
			tempSigma = Math.pow(0.1 + 0.1 * i, 2);
			computeTrueBasedCondProWithInteg(tempSigma);
			for (int j = 0; j < data.LEVEL; j++) {
				for (int k = 0; k < data.LEVEL; k++) {
					trueBasedPros[i][j][k] = trueBasedCondPro[j][k];
				}// Of for k
			}// Of for j

			// printMatrix(trueBasedCondProForSigmas[i]);
		}// Of for i
	}// Of computeTrueBasedIntegrationPro

	/**
	 * Euclidean distance
	 */
	private double euclideanDistance(double[] paraArray1, double[] paraArray2) {
		double tempDistance = 0;

		for (int i = 0; i < paraArray1.length; i++) {
			tempDistance += Math.abs(paraArray1[i] - paraArray2[i]);
		}// of for i
		return tempDistance;
	}// Of euclidenDistance

	/**
	 * Find the optimal sigmas
	 */
	public double[][] findDistributionForOptimalSigmas() {
		double[] tempDistance = new double[trueBasedPros.length];
		double[][] tempOptimalSigmas = new double[numOfGroups][data.LEVEL];
		for (int i = 0; i < numOfGroups; i++) {
			for (int j = 0; j < trueGroupBasedCountPro[0].length; j++) {
				// Step 1. Compute the distances
				for (int k = 0; k < trueBasedPros.length; k++) {
					for (int l = 0; l < trueBasedPros[0].length; l++) {
						tempDistance[k] = euclideanDistance(
								trueGroupBasedCountPro[i][j],
								trueBasedPros[k][l]);
					}// Of for l
				}// Of for k

				// Step 2. Find the distribution of conditional probability for
				// the optimal sigma
				int tempIndex = minValueIndex(tempDistance);
				// Print the optimal sigmas for each level on each group
				tempOptimalSigmas[i][j] = (tempIndex + 1 + 0.0) / 10;
				System.out.println("Group-" + i + " level-" + j + " Sigma is :"
						+ tempOptimalSigmas[i][j]);
				for (int k = 0; k < data.LEVEL; k++) {
					trueGroupBasedCountPro[i][j][k] = trueBasedPros[tempIndex][j][k];
				}// Of for k
			}// Of for j
				// Check
				// System.out.println("The group-" + i
				// +" distribution for the optimal sigmas: ");
			// printMatrix(trueGroupBasedCountPro[i]);
		}// Of for i
		return tempOptimalSigmas;
	}// Of findOptimalSigmas

	/**
	 * The index of minimum value for a given array
	 */
	private int minValueIndex(double[] paraArray) {
		int tempIndex = 0;
		double tempMinValue = 9999;

		for (int i = 0; i < paraArray.length; i++) {
			if (tempMinValue > paraArray[i]) {
				tempMinValue = paraArray[i];
				tempIndex = i;
			}// of if
		}// Of for i
		return tempIndex;
	}// Of minValueIndex

	/**
	 * Compute number of rating
	 */
	public void computeGroupPropOfVisRating() {
		// The degree of a group (e.g.: group-0)
		visGroupDegree = new int[numOfGroups];
		visGroupLevelDegree = new int[numOfGroups][data.LEVEL];
		visPropGroup = new double[numOfGroups][data.LEVEL];

		// Step 1. Count the degree of visible rating for each group
		for (int g = 0; g < numOfGroups; g++) {
			for (int i = 0; i < data.itemNum; i++) {
				// &&&group ratings
				visGroupDegree[g] += itemGroupDegree[g][i];
				for (int j = 0; j < data.LEVEL; j++) {
					visGroupLevelDegree[g][j] += itemGroupLevelDegree[g][i][j];
				}// Of for j

			}// Of for i

			for (int i = 0; i < data.LEVEL; i++) {
				if(visGroupDegree[g] != 0){
					visPropGroup[g][i] = visGroupLevelDegree[g][i] * 1.0
							/ visGroupDegree[g];
				}//Of if
			}// Of for i
		}// Of for j

		System.out.println("The proportions of visible rating: ");
		printMatrix(visPropGroup);
	}// Of computeGroupPropOfVisRating

	/**
	 * Gauss-seidel iteration
	 */
	public void computePropOfTrueRatingGaussSeidel() {
		truePropGroup = new double[numOfGroups][data.LEVEL];
		double[][] groupBasedCondPro2;
		for (int i = 0; i < numOfGroups; i++) {
			groupBasedCondPro2 = transposition(trueGroupBasedCountPro[i]);
			truePropGroup[i] = Iteration.Gauss_seidel(
					groupBasedCondPro2, truePropGroup[i], visPropGroup[i],
					1e-5);
		}// Of for i

		// Step 5. If propOfTrueRating < 0, then propOfTrueRating = 0
		for (int i = 0; i < numOfGroups; i++) {
			for (int j = 0; j < data.LEVEL; j++) {
				if (truePropGroup[i][j] < 1e-6) {
					truePropGroup[i][j] = 0;
				}// Of if
			}// Of for j

			// Step 6. normalization
			double tempTotalProp = 0;
			for (int j = 0; j < data.LEVEL; j++) {
				tempTotalProp += truePropGroup[i][j];
			}// of for j
			for (int j = 0; j < data.LEVEL; j++) {
				truePropGroup[i][j] /= tempTotalProp;
				truePropGroup[i][j] = Math.round(truePropGroup[i][j] * 10000) / 10000.0;
			}// of for j
		}// Of for i

		System.out
				.println("The proportions of true rating (afer normalization): ");
		printMatrix(truePropGroup);
	}// Of computePropOfTrueRatingGaussSeidel

	/**
	 * Employ Bayes method to compute visible-rating based conditional
	 * probability
	 * 
	 */
	public void computeVisBasedCondProBasedOnBayes() {
		visGroupBasedBayesPro = new double[numOfGroups][data.LEVEL][data.LEVEL];

		for (int i = 0; i < numOfGroups; i++) {
			// Step 1. Employ Bayes method to compute conditional probability
			// based
			// on real rating
			for (int j = 0; j < data.LEVEL; j++) {
				for (int k = 0; k < data.LEVEL; k++) {
					if (visPropGroup[i][j] > 1e-6) {
						visGroupBasedBayesPro[i][j][k] = truePropGroup[i][j]
								* trueGroupBasedCountPro[i][k][j]
								/ visPropGroup[i][j];
					}// of if
				}// of for k
			}// Of for i

			System.out.println("visGroupBasedBayesPro before normalization: ");
			printMatrix(visGroupBasedBayesPro[i]);

			// ???归一化
			double tempProbability = 0;
			for (int j = 0; j < data.LEVEL; j++) {
				tempProbability = 0;
				for (int k = 0; k < data.LEVEL; k++) {
					tempProbability += visGroupBasedBayesPro[i][j][k];
				}// Of for k
				for (int k = 0; k < data.LEVEL; k++) {
					visGroupBasedBayesPro[i][j][k] /= tempProbability;
					visGroupBasedBayesPro[i][j][k] = Math
							.round(visGroupBasedBayesPro[i][j][k] * 10000) / 10000.0;
				}// Of for k
			}// Of for j

			System.out.println("visGroupBasedBayesPro after normalization: ");
			printMatrix(visGroupBasedBayesPro[i]);
		}// Of for i
	}// Of computeRealBasedCondProbabilityBasedOnBayes

	/**
	 ************************* 
	 * Print a double array.
	 ************************* 
	 */
	public static void printArray(double[] paraDoubleArray) {
		for (int i = 0; i < paraDoubleArray.length; i++) {
			System.out.print(paraDoubleArray[i] + " ");
		}// Of for i
		System.out.println();
	}// Of printArray

	/**
	 ************************* 
	 * Print a double array.
	 ************************* 
	 */
	public static void printArray(int[] paraDoubleArray) {
		for (int i = 0; i < paraDoubleArray.length; i++) {
			System.out.print(paraDoubleArray[i] + " ");
		}// Of for i
		System.out.println();
	}// Of printArray

	/**
	 *************************** 
	 * Print an int matrix, simply for test.
	 * 
	 * @param paraMatrix
	 *            The given matrix. Different rows may contain different number
	 *            of values.
	 *************************** 
	 */
	public static void printMatrix(int[][] paraMatrix) {
		if (paraMatrix.length == 0) {
			System.out.println("This is an empty matrix.");
			return;
		} else {
			System.out.println("This is an int matrix: ");
		}// Of if

		for (int i = 0; i < paraMatrix.length; i++) {
			for (int j = 0; j < paraMatrix[i].length; j++) {
				System.out.print("" + paraMatrix[i][j] + "\t");
			}// Of for j
			System.out.println();
		}// Of for i
	}// Of printMatrix

	/**
	 *************************** 
	 * Print an int matrix, simply for test.
	 * 
	 * @param paraMatrix
	 *            The given matrix. Different rows may contain different number
	 *            of values.
	 *************************** 
	 */
	public static void printMatrix(double[][] paraMatrix) {
		if (paraMatrix.length == 0) {
			System.out.println("This is an empty matrix.");
			return;
		} else {
			System.out.println("This is an int matrix: ");
		}// Of if

		for (int i = 0; i < paraMatrix.length; i++) {
			for (int j = 0; j < paraMatrix[i].length; j++) {
				System.out.print("" + paraMatrix[i][j] + "\t");
			}// Of for j
			System.out.println();
		}// Of for i
	}// Of printMatrix

	/**
	 * Compute MAE
	 * 
	 */
	public double computeMG() {
		double[] tempGroupMG = new double[numOfGroups];
		double tempMG = 0;
		for (int g = 0; g < numOfGroups; g++) {
			for (int i = 0; i < data.LEVEL; i++) {
				for (int j = 0; j < data.LEVEL; j++) {
					if (trueGroupLevelDegree[g][i] > 1e-6) {
						tempGroupMG[g] += visGroupBasedBayesPro[g][i][j]
								* trueGroupLevelDegree[g][i] * Math.abs(j - i);
					}// Of if
				}// of for j
			}// of for i

			if(visGroupDegree[g] != 0){
				tempGroupMG[g] = tempGroupMG[g] / visGroupDegree[g];
			}//Of if

			tempMG += tempGroupMG[g] * visGroupDegree[g];
		}// Of for g

		tempMG = tempMG / visTotalDegree;

		return tempMG;
	}// of computeMAE

	public static void main(String[] args) {
		try {
			// TODO 自动生成的方法存根
			String tempPropertyFileName = new String("src/properties/ml-100k.properties");
			DataInfo tempData = new DataInfo(tempPropertyFileName);
			tempData.readData();
			tempData.computeAverageRating();
			LevelGroupMagicBarrier tempMagic = new LevelGroupMagicBarrier(
					tempData, 4); 
			tempMagic.computeTrueGroupCountPro();
			tempMagic.findDistributionForOptimalSigmas();
			tempMagic.computeGroupPropOfVisRating();
			tempMagic.computePropOfTrueRatingGaussSeidel();
			tempMagic.computeVisBasedCondProBasedOnBayes();
			double tempMG = tempMagic.computeMG();
			System.out.println("MG: " + tempMG);
		} catch (Exception e) {
			e.printStackTrace();
		}// Of try
	}// Of main
}// Of class MagicOfUniqueMovieLens943u1682m