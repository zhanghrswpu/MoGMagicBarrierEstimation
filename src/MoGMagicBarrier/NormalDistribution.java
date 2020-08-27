package MoGMagicBarrier;

public class NormalDistribution {
	/**
	 * ÕýÌ¬·Ö²¼£º f(x) = 1 / (sqrt(2 * PI) * Sigma) * (e ^(-(x - mu)^ 2) / 2 * pow(Sigma,
	 * 2)))
	 */
	public static double normrnd(double paraMu, double paraSigma, double paraX) {
		double tempNormP = 0;

		tempNormP = 1 / (Math.sqrt(2 * Math.PI) * paraSigma)
				* Math.pow(Math.E, (-Math.pow((paraX - paraMu) / paraSigma, 2) / 2));
		return tempNormP;
	}// of normrnd

	/**
	 * Step-length integration
	 * F(x) =\int_{paraA}^{paraB} 1 / (sqrt(2 * PI) * paraSigma) *
	 *  (e ^(-(x - paraMu)^ 2) / 2 * pow(paraSigma, 2))) dx
	 */
	public static double stepLengthIntegration(double paraMu, double paraSigma, double paraA, double paraB) {
		int tempLength = 10000;
		double tempStepLength = (paraB - paraA) / tempLength;
		double tempStepIntegration = 0;

		for (int i = 0; i < tempLength; i++) {
			tempStepIntegration += tempStepLength
					* normrnd(paraMu, paraSigma, paraA + (2 * i + 1) * tempStepLength / 2);
		} // of for i

		return tempStepIntegration;
	}// of stepLengthIntegration
}// of class NormalDistribution
