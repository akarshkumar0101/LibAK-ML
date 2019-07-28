//package machinelearning.neuralnet;
//
//public class SynchronizedNeuralNetwork extends NeuralNetwork {
//
//	@Override
//	public synchronized void initNetworkAndZ(double[][] nodes) {
//		super.initNetworkAndZ(nodes);
//	}
//
//	@Override
//	public synchronized void initWeights(double[][][] weights) {
//		super.initWeights(weights);
//	}
//
//	@Override
//	public synchronized void initBiases(double[][] biases) {
//		super.initBiases(biases);
//	}
//
//	@Override
//	protected synchronized void initPartials(boolean init) {
//		super.initPartials(init);
//	}
//
//	@Override
//	public synchronized Object exportParameters() {
//		return super.exportParameters();
//	}
//
//	@Override
//	public synchronized void feed(double... inputLayerData) {
//		super.feed(inputLayerData);
//	}
//
//	@Override
//	public synchronized double[] getOutput() {
//		return super.getOutput();
//	}
//
//	@Override
//	public synchronized void randomizeWeightsAndBiases() {
//		super.randomizeWeightsAndBiases();
//	}
//
//	@Override
//	public synchronized void randomizeWeightsAndBiases(double lowlw, double uplw, double lowlb, double uplb) {
//		super.randomizeWeightsAndBiases(lowlw, uplw, lowlb, uplb);
//	}
//
//	@Override
//	public synchronized void calculateFully() {
//		super.calculateFully();
//	}
//
//	@Override
//	protected synchronized void calculateLayer(int layer, boolean recurseFully) {
//		super.calculateLayer(layer, recurseFully);
//	}
//
//	// TRAINING
//	@Override
//	public synchronized void setTrainer(OldNeuralNetworkTrainer<?, ?> trainer) {
//		super.setTrainer(trainer);
//	}
//
//	@Override
//	protected synchronized void gradientDescent(double[][][] examples) {
//		super.gradientDescent(examples);
//	}
//
//	@Override
//	protected synchronized void calculatePartials(double[] expectedOutput, boolean recurseFully) {
//		super.calculatePartials(expectedOutput, recurseFully);
//	}
//
//	@Override
//	protected synchronized void calculatePartials(int layer, boolean recurseFully) {
//		super.calculatePartials(layer, recurseFully);
//	}
//
//	@Override
//	protected synchronized void takeWeightGradientDescentStep(double[][][] dW) {
//		super.takeWeightGradientDescentStep(dW);
//	}
//
//	@Override
//	protected synchronized void takeBiasGradientDescentStep(double[][] dB) {
//		super.takeBiasGradientDescentStep(dB);
//	}
//
//	/**
//	 * trainingExamples[trainingExampleNum][0] is the input layer
//	 * trainingExamples[trainingExampleNum][1] is the expected output layer
//	 *
//	 * @param trainingExamples
//	 * @return the average cost of the network and the average accuracy in
//	 *         classification
//	 */
//
//	@Override
//	public synchronized double[] calculateAveragePerformance(double[][][] trainingExamples) {
//		return super.calculateAveragePerformance(trainingExamples);
//	}
//
//	/**
//	 * trainingExamples[0] is the input layer
//	 *
//	 * trainingExamples[1] is the expected output layer
//	 *
//	 * @param trainingExamples
//	 * @return the cost of the network ("how bad it is") and boolean if it was
//	 *         correct result
//	 */
//	@Override
//	public synchronized Object[] calculatePerformance(double[][] trainingExample) {
//		return super.calculatePerformance(trainingExample);
//	}
//
//	@Override
//	public synchronized String toString() {
//		return super.toString();
//	}
//
//}
