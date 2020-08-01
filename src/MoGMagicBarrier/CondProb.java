package MoGMagicBarrier;

import algorithm.MLGMDN;
import datamodel.DataInfo;
import tool.SimpleTool;

public class CondProb {
	public DataInfo data;

	/**
	 * Sigma
	 */
	public float[] sigma;

	/**
	 * Conditional probability based “true” rating For example: 3 * 5 * 5 (numNoise
	 * * LEVEL * LEVEL)
	 */
	double[][][] trueBasedCondPro;

	/**
	 * To count rating distribution, we need to obtain the rating number of each
	 * scale.
	 */

	public double[] totalGammaOfVisRating; // numNoise
	public double[][] gammaOfEachVisRating; // numNoise * LEVEL
	public double[][] propOfEachVisRating; // numNoise * LEVEL
	public double[][] propOfTrueRating; // numNoise * LEVEL

	/**
	 * Conditional probability based "visible" rating
	 */
	public double[][][] visBasedCondPro; // numNoise * LEVEL * LEVEL

	/**
	 * Constructor
	 */
	public CondProb(DataInfo paraData) {
		data = paraData;
		trueBasedCondPro = new double[data.numNoise][data.LEVEL][data.LEVEL];
		visBasedCondPro = new double[data.numNoise][data.LEVEL][data.LEVEL];
		totalGammaOfVisRating = new double[data.numNoise];
		gammaOfEachVisRating = new double[data.numNoise][data.LEVEL];
		propOfEachVisRating = new double[data.numNoise][data.LEVEL];
		propOfTrueRating = new double[data.numNoise][data.LEVEL];
	}// of the first constructor

	/**
	 * Get index based on the rating
	 */
	public int getIndexBasedRating(double paraRating) {
		int tempIndex = 0;

		for (int i = 0; i < data.LEVEL; i++) {
			double tempLow = data.rlow + i * data.stepLen - data.stepLen / 2;
			double tempHigh = data.rlow + i * data.stepLen + data.stepLen / 2;

			if (paraRating > tempLow && paraRating <= tempHigh) {
				tempIndex = i;
				break;
			} // of if
		} // of for i

		return tempIndex;
	}// of getIndexBasedRating

	/**
	 * Compute true-rating-based condition probability with step-length integration
	 * 
	 * output: trueBasedCondPro (numNoise * LEVEL * LEVEL)
	 */
	public double[][][] computeTrueBasedCondProWithInteg(float[] paraSigma) {
		sigma = new float[paraSigma.length];
		for (int i = 0; i < paraSigma.length; i++) {
			sigma[i] = (float) Math.sqrt(paraSigma[i]);
		} // of for i

		for (int k = 0; k < data.numNoise; k++) {
			for (int z = 0; z < data.LEVEL; z++) {
				// stepLengthIntegration(i + 1, paraSigma, -1000, 0)
				for (int y = 0; y < data.LEVEL; y++) {
					/*
					 * if (y == 0) { trueBasedCondPro[k][y][z] =
					 * NormalDistribution.stepLengthIntegration(rlow + z * stepLen, sigma[k], -1000,
					 * rlow + stepLen / 2);//1.5 } else if (y == LEVEL - 1) {
					 * trueBasedCondPro[k][y][z] = NormalDistribution.stepLengthIntegration(rlow + z
					 * * stepLen, sigma[k], rhigh - stepLen / 2, 1006); } else {
					 * trueBasedCondPro[k][y][z] = NormalDistribution.stepLengthIntegration(rlow + z
					 * * stepLen, sigma[k], rlow + y * stepLen - stepLen / 2, rlow + y * stepLen +
					 * stepLen / 2); //? } // Of if
					 */
					trueBasedCondPro[k][y][z] = NormalDistribution.stepLengthIntegration(
							data.rlow + z * data.stepLen, sigma[k],
							data.rlow + y * data.stepLen - data.stepLen / 2, 
							data.rlow + y * data.stepLen + data.stepLen / 2); // ?
				} // Of for j
			} // Of for i
		} // of for k

		// ???归一化
		double tempTotalProbability = 0;
		for (int k = 0; k < data.numNoise; k++) {
			for (int z = 0; z < data.LEVEL; z++) {
				tempTotalProbability = 0;
				for (int y = 0; y < data.LEVEL; y++) {
					tempTotalProbability += trueBasedCondPro[k][y][z];
				} // Of for j
					// System.out.println(i + "-tempTotalProbability: " +
					// tempTotalProbability);
				for (int y = 0; y < data.LEVEL; y++) {
					trueBasedCondPro[k][y][z] /= tempTotalProbability;
					trueBasedCondPro[k][y][z] = Math.round(trueBasedCondPro[k][y][z] * 10000) / 10000.0;
				} // Of for j
			} // Of for i
		} // of for k

		return trueBasedCondPro;
	}// Of computeTrueBasedCondProWithInteg

	/**
	 * Compute probability of visual rating
	 * 
	 * output: propOfEachVisRating (numNoise * LEVEL)
	 */
	public void computePropOfVisRating(MLGMDN paraMLG) {
		// System.out.println(paraData.trainVector + "test 0");
		// SimpleTool.printDoubleArray(paraData.trainVector);
		// System.out.println(paraData.trainVector + "test 1");
		// SimpleTool.printFloatArray(paraData.trainVector);
		// SimpleTool.printMatrix(paraMLG.noiseDistribution);
		// System.out.println("rlow: " + rlow + " stepLen: " + stepLen);
		for (int i = 0; i < data.numOfRatings; i++) {
			double tempOrigRating = data.dataVector[i];
			for (int k = 0; k < data.numNoise; k++) {
				totalGammaOfVisRating[k] += paraMLG.noiseDistribution[i][k];
				for (int y = 0; y < data.LEVEL; y++) {
					if (tempOrigRating > data.rlow + y * data.stepLen - data.stepLen / 2
							&& tempOrigRating <= data.rlow + y * data.stepLen + data.stepLen / 2) {
						gammaOfEachVisRating[k][y] += paraMLG.noiseDistribution[i][k];
					} // Of if
				} // of for k
			} // of for j
		} // of for i

		for (int k = 0; k < data.numNoise; k++) {
			for (int y = 0; y < data.LEVEL; y++) {
				propOfEachVisRating[k][y] = gammaOfEachVisRating[k][y] / totalGammaOfVisRating[k];
			} // of for j
		} // Of for i

		// System.out.println("The proportions of visible rating: ");
		// printDoubleArray(propOfVisRating);
	}// Of computeNumberOfRating

	/**
	 * transposition
	 */
	public double[][] transposition(double[][] paraMatrix) {
		double[][] tempMatrix = new double[paraMatrix[0].length][paraMatrix.length];
		for (int i = 0; i < paraMatrix.length; i++) {
			for (int j = 0; j < paraMatrix[0].length; j++) {
				tempMatrix[j][i] = paraMatrix[i][j];
			} // Of for j
		} // Of for i
		return tempMatrix;
	}// Of transposition

	/**
	 * Gauss-seidel iteration
	 * 
	 * output: propOfTrueRating (numNoise * LEVEL)
	 */
	public void computePropOfTrueRatingSOR() {
		
		for (int k = 0; k < data.numNoise; k++) {
			for (int z = 0; z < data.LEVEL; z++) {
				propOfTrueRating[k][z] = 0.2;
			} // Of if
		} // of for k

		// double[][][] trueBasedCondPro2 = new double[numNoise][][];
		for (int k = 0; k < data.numNoise; k++) {
			// trueBasedCondPro2[k] = transposition(trueBasedCondPro[k]);
			// input: trueBasedCondPro[k], LEVEL * LEVEL
			// input: propOfEachVisRating[k], LEVEL
			// output: propOfTrueRating[k], LEVEL
			// propOfTrueRating[k] = Iteration.SOR(trueBasedCondPro[k],
			// propOfEachVisRating[k], 0.1, 1e-3);
			// propOfTrueRating[k] =
			// GuassSeidelInteration.Gauss_seidel2(trueBasedCondPro[k],
			// propOfTrueRating[k], propOfEachVisRating[k], 1e-5);
			propOfTrueRating[k] = Iteration.SOR(trueBasedCondPro[k], propOfEachVisRating[k], 0.5, 1e-5);
		} // of for k

		// Step 4. Check
		double[][] tempPropOfVisibleRating = new double[data.numNoise][data.LEVEL];
		for (int k = 0; k < data.numNoise; k++) {
			for (int z = 0; z < data.LEVEL; z++) {
				for (int y = 0; y < data.LEVEL; y++) {
					tempPropOfVisibleRating[k][y] += propOfTrueRating[k][z] * trueBasedCondPro[k][y][z];
				} // Of for j
			} // Of for i
		} // of for k

		// Step 5. If propOfTrueRating < 0, then propOfTrueRating = 1e-4
		for (int k = 0; k < data.numNoise; k++) {
			for (int z = 0; z < data.LEVEL; z++) {
				if (propOfTrueRating[k][z] < 1e-6) {
					propOfTrueRating[k][z] = 1e-4;
				} // Of if
			} // Of for i
		} // of for k

		// Step 6. normalization
		double tempTotalProp = 0;
		for (int k = 0; k < data.numNoise; k++) {
			tempTotalProp = 0;
			for (int z = 0; z < data.LEVEL; z++) {
				tempTotalProp += propOfTrueRating[k][z];
			} // of for i
			for (int z = 0; z < data.LEVEL; z++) {
				propOfTrueRating[k][z] /= tempTotalProp;
				propOfTrueRating[k][z] = Math.round(propOfTrueRating[k][z] * 10000) / 10000.0;
			} // of for i

			// System.out
			// .println("The proportions of true rating (afer normalization):
			// ");
			// printDoubleArray(propOfTrueRating);

			// Step 7. Check 2
			for (int y = 0; y < data.LEVEL; y++) {
				tempPropOfVisibleRating[k][y] = 0;
				for (int z = 0; z < data.LEVEL; z++) {
					tempPropOfVisibleRating[k][y] += propOfTrueRating[k][z] * trueBasedCondPro[k][y][z];
				} // Of for j
			} // Of for i
		} // of for k
		System.out.println("---------------");
		SimpleTool.printMatrix(propOfEachVisRating);
		SimpleTool.printMatrix(tempPropOfVisibleRating);
		System.out.println("---------------");
	}// Of computePropOfTrueRatingGaussSeidel

	/**
	 * Employ Bayes method to compute visible-rating based conditional probability
	 * 
	 */
	public void computeVisBasedCondProBasedOnBayes() {
		// Step 1. Employ Bayes method to compute conditional probability based
		// on real rating
		for (int k = 0; k < data.numNoise; k++) {
			for (int y = 0; y < visBasedCondPro[0].length; y++) {
				for (int z = 0; z < visBasedCondPro[0][0].length; z++) {
					visBasedCondPro[k][z][y] = propOfTrueRating[k][z] * trueBasedCondPro[k][y][z]
							/ propOfEachVisRating[k][y];// switching
														// problem?:
														// i,j
				} // of for j
			} // Of for i
				// SimpleTool.printDoubleArray(propOfTrueRating[k]);
				// SimpleTool.printMatrix(visBasedCondPro[k]);
		} // of for k

		// ???归一化
		double tempProbability = 0;
		for (int k = 0; k < data.numNoise; k++) {
			for (int y = 0; y < data.LEVEL; y++) {
				tempProbability = 0;
				for (int z = 0; z < data.LEVEL; z++) {
					tempProbability += visBasedCondPro[k][z][y];
				} // Of for j
				for (int z = 0; z < data.LEVEL; z++) {
					visBasedCondPro[k][z][y] /= tempProbability;
					visBasedCondPro[k][z][y] = Math.round(visBasedCondPro[k][z][y] * 10000) / 10000.0;
				} // Of for j
			} // Of for i
		} // of for k
	}// Of computeRealBasedCondProbabilityBasedOnBayes
}// of class ConditionalProbability
