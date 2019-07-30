package machinelearning.neuralnet;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;

import javax.swing.JComponent;

import machinelearning.neuralnet.FCNeuralNetwork;

public class VisualNetworkPanel extends JComponent {

	private static final long serialVersionUID = 3090146802464835357L;

	private FCNeuralNetwork network;
	private double[] expectedOutput;

	public VisualNetworkPanel(FCNeuralNetwork network) {
		super();
		this.setNetwork(network);

	}

	public void setNetwork(FCNeuralNetwork network) {
//		if (this.network != null) {
//			this.network.removeNeuralNetworkListener(this);
//		}
//		this.network = network;
//		this.network.addNeuralNetworkListener(this);
		this.network = network;
	}

	public FCNeuralNetwork getNetwork() {
		return this.network;
	}

	@Override
	public void paintComponent(Graphics g) {
		super.paintComponent(g);
		g.setColor(Color.LIGHT_GRAY);
		g.fillRect(0, 0, this.getWidth(), this.getHeight());
		g.setColor(Color.DARK_GRAY);
		for (int layer = 0; layer < this.network.networkDimensions.length; layer++) {
			int circledia = (int) Math.max(10,
					Math.min(this.getWidth() / 1.5 / (this.network.networkDimensions.length + 1),
							this.getHeight() / 1.5 / this.network.networkDimensions[layer]));
			for (int nodeID = 0; nodeID < this.network.networkDimensions[layer]; nodeID++) {
				double value = this.network.nodes[layer][nodeID];
				g.setColor(new Color((int) (value * 255), (int) (value * 255), (int) (value * 255)));
				int[] loc = this.locationOfNode(layer, nodeID);
				g.fillOval(loc[0], loc[1], circledia, circledia);
			}
		}
		for (int layer = 1; layer < this.network.networkDimensions.length; layer++) {
			for (int node2ID = 0; node2ID < this.network.networkDimensions[layer]; node2ID++) {
				for (int node1ID = 0; node1ID < this.network.networkDimensions[layer - 1]; node1ID++) {
					if (this.network.networkDimensions[layer] * this.network.networkDimensions[layer - 1] > 10000
							&& Math.random() > 1) {
						continue;
					}

					double weight = this.network.weights[layer - 1][node2ID][node1ID];
					int[] loc1 = this.centerLocationOfNode(layer - 1, node1ID);
					int[] loc2 = this.centerLocationOfNode(layer, node2ID);

					Graphics2D g2 = (Graphics2D) g;
					g2.setStroke(new BasicStroke((float) Math.abs(weight)));
					g2.setColor(weight >= 0 ? Color.GREEN : Color.RED);

					if (Math.abs(weight) > .01) {
						g2.drawLine(loc1[0], loc1[1], loc2[0], loc2[1]);
					}
				}

			}
		}
		if (this.expectedOutput != null) {
			int circledia = (int) Math.max(10,
					Math.min(this.getWidth() / 1.5 / (this.network.networkDimensions.length + 1), this.getHeight() / 1.5
							/ this.network.networkDimensions[this.network.networkDimensions.length - 1]));
			for (int nodeID = 0; nodeID < this.expectedOutput.length; nodeID++) {
				double value = this.expectedOutput[nodeID];
				g.setColor(new Color((int) (value * 255), (int) (value * 255), (int) (value * 255)));
				int[] loc = new int[] { (int) (this.getWidth() - circledia * 1.2),
						(int) this.scale(nodeID, 0, this.expectedOutput.length, 0, this.getHeight()) };
				g.fillOval(loc[0], loc[1], circledia, circledia);
			}
		}
	}

	private int[] centerLocationOfNode(int layer, int nodeID) {
		int circledia = (int) Math.max(10, Math.min(this.getWidth() / 1.5 / (this.network.networkDimensions.length + 1),
				this.getHeight() / 1.5 / this.network.networkDimensions[layer]));
		int[] loc = this.locationOfNode(layer, nodeID);
		loc[0] += circledia / 2;
		loc[1] += circledia / 2;
		return loc;
	}

	private int[] locationOfNode(int layer, int nodeID) {
		int x = (int) this.scale(layer, 0, this.network.networkDimensions.length + 1, 0, this.getWidth());
		int y = (int) this.scale(nodeID, 0, this.network.networkDimensions[layer], 0, this.getHeight());
		return new int[] { x, y };
	}

	private double scale(double num, double s1low, double s1up, double s2low, double s2up) {
		num -= s1low;
		num *= (s2up - s2low) / (s1up - s1low);
		num += s2low;
		return num;
	}

}
