package machine.learning.demo;

import java.util.Random;

import javax.swing.JFrame;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.ThresholdCurve;
import weka.classifiers.trees.J48;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.gui.treevisualizer.PlaceNode2;
import weka.gui.treevisualizer.TreeVisualizer;
import weka.gui.visualize.PlotData2D;
import weka.gui.visualize.ThresholdVisualizePanel;
import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;

public class PredictAnimal {

	public static void main(String[] args) throws Exception {


		DataSource source = new DataSource("C:\\Users\\Dung Dinh\\eclipse-workspace\\demo\\src\\main\\resources\\zoo.arff");
		Instances data = source.getDataSet();
		System.out.println(data.numInstances() + " instances loaded.");
		String[] opts = new String[] { "-R", "1" };
		Remove remove = new Remove();
		remove.setOptions(opts);
		remove.setInputFormat(data);
		data = Filter.useFilter(data, remove);
		AttributeSelection attSelect = new AttributeSelection();
		InfoGainAttributeEval eval = new InfoGainAttributeEval();
		Ranker search = new Ranker();
		attSelect.setEvaluator(eval);
		attSelect.setSearch(search);
		attSelect.SelectAttributes(data);
		int[] indices = attSelect.selectedAttributes();

		/*
		 * Build a decision tree
		 */
		String[] options = new String[1];
		options[0] = "-U";
		J48 tree = new J48();
		tree.setOptions(options);
		tree.buildClassifier(data);

		/*
		 * Classify new instance.
		 */
		double[] vals = new double[data.numAttributes()];
		vals[0] = 1.0; // hair {false, true}
		vals[1] = 0.0; // feathers {false, true}
		vals[3] = 1.0; // airborne {false, true}
		vals[2] = 0.0; // eggs {false, true}
		vals[3] = 1.0; // milk {false, true}
		vals[4] = 0.0; // airborne {false, true}
		vals[5] = 0.0; // aquatic {false, true}
		vals[6] = 0.0; // predator {false, true}
		vals[7] = 1.0; // toothed {false, true}
		vals[8] = 1.0; // backbone {false, true}
		vals[9] = 1.0; // breathes {false, true}
		vals[10] = 1.0; // venomous {false, true}
		vals[11] = 0.0; // fins {false, true}
		vals[12] = 4.0; // legs INTEGER [0,9]
		vals[13] = 1.0; // tail {false, true}
		vals[14] = 1.0; // domestic {false, true}
		vals[15] = 0.0; // catsize {false, true}
		Instance animal = new DenseInstance(1.0, vals);
		animal.setDataset(data); 

		double label = tree.classifyInstance(animal);
		System.out.println("That is: " + data.classAttribute().value((int) label).toUpperCase());

	}
}
