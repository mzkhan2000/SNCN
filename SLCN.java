package classifiers;

import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Set;

import mulan.classifier.InvalidDataException;
import mulan.classifier.ModelInitializationException;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;
import mulan.data.LabelNode;
import mulan.data.LabelsMetaData;
import mulan.data.MultiLabelInstances;
import mulan.dimensionalityReduction.BinaryRelevanceAttributeEvaluator;

import org.apache.commons.lang.ArrayUtils;

import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.AttributeEvaluator;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;
import weka.filters.unsupervised.attribute.Remove;

/**
 * 
 * @author andeoliv
 *
 */
public class SLCN implements MultiLabelLearner{

	protected LabelsMetaData metadata;
	
	protected MultiLabelInstances data;
	
	protected AbstractClassifier classifier;
	protected Map<Integer,FilteredClassifier> classifiers;
	protected Map<Integer,Integer> labelIndices;
	
	protected int maxLocalAttributes = Integer.MAX_VALUE;
	protected int maxGlobalAttributes = Integer.MAX_VALUE;
	protected int maxTrainSampleSize = Integer.MAX_VALUE;
	
	protected boolean debug = false;
	protected double biasToUniformClass = 0.0;
	
	protected Remove globalFilter = null;
	
	protected ASEvaluation fsMetric = new InfoGainAttributeEval();
	
	protected static String norm = "none";
	protected static String mode = "eval";
	protected static String combapp = "avg";
	
	public SLCN() throws FileNotFoundException {
		this(new J48());
	}
	
	public SLCN(AbstractClassifier classifier) throws FileNotFoundException {
		this.classifier = classifier;
	}
	
	public SLCN(AbstractClassifier classifier, int maxAttributes, int maxTrainSampleSize) throws FileNotFoundException {
		this(classifier);
		this.maxLocalAttributes = maxAttributes;
		this.maxTrainSampleSize = maxTrainSampleSize;
	}
	
	public SLCN(AbstractClassifier classifier, int maxLocalAttributes, int maxGlobalAttributes, int maxTrainSampleSize) throws FileNotFoundException {
		this(classifier);
		this.maxLocalAttributes = maxLocalAttributes;
		this.maxGlobalAttributes = maxGlobalAttributes;
		this.maxTrainSampleSize = maxTrainSampleSize;
	}
	
	public SLCN(AbstractClassifier classifier, int maxLocalAttributes, int maxTrainSampleSize, double biasToUniformClass) throws FileNotFoundException {
		this(classifier, maxLocalAttributes, maxTrainSampleSize);
		setBiasToUniformClass(biasToUniformClass);
	}
	
	public SLCN(AbstractClassifier classifier, int maxLocalAttributes, int maxGlobalAttributes,int maxTrainSampleSize, double biasToUniformClass) throws FileNotFoundException {
		this(classifier, maxLocalAttributes, maxGlobalAttributes, maxTrainSampleSize);
		setBiasToUniformClass(biasToUniformClass);
	}
	
	protected MultiLabelInstances globalFilter(MultiLabelInstances mulanData, int maxAttributes) throws Exception {
		
		BinaryRelevanceAttributeEvaluator fs = new BinaryRelevanceAttributeEvaluator(fsMetric, mulanData, combapp, norm, mode);
	
		final Map<Integer, Double> infogains = new HashMap<Integer, Double>();
		for (int att=0; att<mulanData.getFeatureIndices().length; att++) {
			double ig = fs.evaluateAttribute(att);
			infogains.put(mulanData.getFeatureIndices()[att], ig);
		}
		
        List<Integer> attributesSortedByGain = new ArrayList<Integer>(infogains.keySet());
        
        Collections.sort(attributesSortedByGain, new Comparator<Integer>(){
       	public int compare(Integer e1, Integer e2) {
                return -Double.compare(infogains.get(e1),infogains.get(e2));
            }
        });
        
        List<Integer> tobeKept = new ArrayList<Integer>();
        for (int i:mulanData.getLabelIndices())
        	tobeKept.add(i);
        
        int count = 0;
        
		for (int i: attributesSortedByGain) {
        	if (count++ < maxAttributes)
        		tobeKept.add(i);
        	else
        		break;
        }
		
		globalFilter = new Remove();
	    
		int[] indices = ArrayUtils.toPrimitive(tobeKept.toArray(new Integer[tobeKept.size()]));
		globalFilter.setInvertSelection(true);
		globalFilter.setAttributeIndicesArray(indices);
		
		globalFilter.setInputFormat(mulanData.getDataSet());

		Instances filteredInstances = Filter.useFilter(mulanData.getDataSet(), globalFilter);
		
		return new MultiLabelInstances(filteredInstances, mulanData.getLabelsMetaData());
	}
	
	protected Filter localFilter(Instances subset, Set<Integer> featuresSet,  int classID) throws Exception {
		
		List<Integer> tobeKept = new LinkedList<Integer>();		

		tobeKept.add(classID);

		subset.setClassIndex(classID);

		if (maxLocalAttributes < featuresSet.size()) { 			

			ASEvaluation evaluator = fsMetric;
			evaluator.buildEvaluator(subset);
			final Map<Integer, Double> infogains = new HashMap<Integer, Double>();
			for (int att: featuresSet) {
				double ig = ((AttributeEvaluator)evaluator).evaluateAttribute(att);
				infogains.put(att, ig);
			}
			
	        List<Integer> attributesSortedByGain = new ArrayList<Integer>(infogains.keySet());
	        
	        Collections.sort(attributesSortedByGain, new Comparator<Integer>(){
	       	public int compare(Integer e1, Integer e2) {
	                return -Double.compare(infogains.get(e1),infogains.get(e2));
	            }
	        });
	        
	        int count = 0;
	        for (int i: attributesSortedByGain) {
	        	if (count++ < maxLocalAttributes)
	        		tobeKept.add(i);
	        	else
	        		break;
	        }
		} else {
			tobeKept.addAll(featuresSet);	
		}
        
        Remove filter = new Remove();
        
		int[] indices = ArrayUtils.toPrimitive(tobeKept.toArray(new Integer[tobeKept.size()]));
		filter.setInvertSelection(true);
		filter.setAttributeIndicesArray(indices);
		filter.setInputFormat(subset);
		
		return filter;
	}
	
	protected Instances sampleInstances(Instances dataset, int sampleSize, double biasToUniformClass) throws Exception {
		
		if (sampleSize>=dataset.numInstances())
			return dataset;
		
		double datasetSize = dataset.numInstances();
		double sizePercent = (100.0*sampleSize)/(datasetSize);
		
		Resample sampler = new Resample();
		sampler.setRandomSeed(0);
		sampler.setSampleSizePercent(sizePercent);
		sampler.setBiasToUniformClass(biasToUniformClass);
		sampler.setInputFormat(dataset);

		Instances sample = Filter.useFilter(dataset, sampler);
		
		return sample; 
		
	}
	
	protected Instances selectInstances(Instances dataset, LabelNode node) {
		LabelNode parentNode = node.getParent();
		
		if (parentNode==null)
			return dataset;

		int superAtt = data.getDataSet().attribute(parentNode.getName()).index();
		
		Instances subset = new Instances(dataset, 0);
		
		for (int i=0; i<dataset.numInstances(); i++) {
			Instance instance = dataset.instance(i);
			if (instance.value(superAtt)==1.0) {
				subset.add(instance);
			}
		}
		
		return subset;
	}

	protected void buildInternal(Instances instances, Set<Integer> featuresSet, LabelNode node) throws Exception,InvalidDataException {
		
		int currClassID = instances.attribute(node.getName()).index();
		String currClassName = node.getName();
		
		if (node.getChildren()!=null && !node.getChildren().isEmpty()) {
			Instances subset = selectInstances(instances, node.getChildren().iterator().next());
			if (subset.numInstances()>0) {
				for (LabelNode childLabelNode: node.getChildren()) {
					buildInternal(subset, featuresSet, childLabelNode);	
				}
			}
		}
		
		Classifier classClassifier = AbstractClassifier.makeCopy(classifier);	
		
		instances.setClassIndex(currClassID);
		
		Instances trainingSet = sampleInstances(instances, maxTrainSampleSize, biasToUniformClass);
		instances = null;
					
		Filter filter = localFilter(trainingSet, featuresSet, currClassID);
		filter.setInputFormat(trainingSet);
		FilteredClassifier classFilteredClassifier = new FilteredClassifier();
		classFilteredClassifier.setClassifier(classClassifier);
		classFilteredClassifier.setFilter(filter);
		trainingSet = Filter.useFilter(trainingSet, filter);
		
		trainingSet.setClass(trainingSet.attribute(currClassName));
		
		classFilteredClassifier.getClassifier().buildClassifier(trainingSet);
		trainingSet = null;
		
		classifiers.put(currClassID, classFilteredClassifier);
	}
		

	public void build(MultiLabelInstances instances) throws Exception,
			InvalidDataException {
		
		if (maxGlobalAttributes<instances.getFeatureIndices().length) 
			instances = globalFilter(instances, maxGlobalAttributes);
		
		data = instances;
		metadata = instances.getLabelsMetaData();		
		
		classifiers = new HashMap<Integer, FilteredClassifier>();
		labelIndices = new HashMap<Integer,Integer>(); 
		
		for (int i=0; i<instances.getLabelIndices().length; i++) 
			labelIndices.put(instances.getLabelIndices()[i], i);
		
		Set<Integer> featuresSet = new HashSet<Integer>();
		for (Attribute att: data.getFeatureAttributes()) 
			featuresSet.add(att.index());
		
		for (LabelNode node: metadata.getRootLabels())
			buildInternal(instances.getDataSet(), featuresSet, node);
		
	}

	public MultiLabelOutput makePrediction(Instance instance) throws Exception,
			InvalidDataException, ModelInitializationException {
		
		if (globalFilter!=null) {
			globalFilter.input(instance);
			instance = globalFilter.output();
		}
		
		
		boolean[] bipartition = new boolean[labelIndices.size()];
		Arrays.fill(bipartition, false);	

		Set<LabelNode> traversedLabelNodes = new HashSet<LabelNode>();
		Queue<LabelNode> breadthFirstLabelNodesQueue = new LinkedList<LabelNode>();
		breadthFirstLabelNodesQueue.addAll(metadata.getRootLabels());
		
		while (!breadthFirstLabelNodesQueue.isEmpty()) {
			
			LabelNode classLabelNode = breadthFirstLabelNodesQueue.poll();
			
			if (!traversedLabelNodes.contains(classLabelNode)) {
				
				int currClassID = data.getDataSet().attribute(classLabelNode.getName()).index();
				
				FilteredClassifier filterdClassClassifier = classifiers.get(currClassID);
				if (filterdClassClassifier!=null) {
					Filter filter = filterdClassClassifier.getFilter();
					Classifier classClassifier = filterdClassClassifier.getClassifier();
					
					filter.input(instance);
					Instance instanceSubset = filter.output();
					
					if (classClassifier.classifyInstance(instanceSubset)==1.0) {
						if (labelIndices.containsKey(currClassID)) {
							bipartition[labelIndices.get(currClassID)] = true;
							
							if (classLabelNode.getChildren()!=null) 
								breadthFirstLabelNodesQueue.addAll(classLabelNode.getChildren());
						}
					} 
					
					traversedLabelNodes.add(classLabelNode);
				}
			}
		}
		
		MultiLabelOutput output = new MultiLabelOutput(bipartition);
		
		return output;
	}
	

	public boolean isUpdatable() {
		return false;
	}

	public MultiLabelLearner makeCopy() throws Exception {
		SLCN copy = new SLCN(classifier, maxLocalAttributes, maxGlobalAttributes, maxTrainSampleSize, biasToUniformClass);
		copy.setFeatureSelectionMetric(fsMetric);
		return copy;
	}

	public void setDebug(boolean arg0) {
		this.debug = arg0;		
	}
	
	public void setMaxAttributes(int maxAttributes) {
		this.maxLocalAttributes = maxAttributes;
	}
		
	public void setMaxTrainSampleSize(int maxTrainSampleSize) {
		this.maxTrainSampleSize = maxTrainSampleSize;
	}
	
	public double getBiasToUniformClass() {
		return biasToUniformClass;
	}
	
	public int getMaxLocalAttributes() {
		return maxLocalAttributes;
	}
	
	public int getMaxGlobalAttributes() {
		return maxGlobalAttributes;
	}
		
	public int getMaxTrainSampleSize() {
		return maxTrainSampleSize;
	}
	
	public Classifier getLocalClassifier() {
		return classifier;
	}
	
	public void setFeatureSelectionMetric(ASEvaluation fsMetric) {
		this.fsMetric = fsMetric;
	}
	
	public ASEvaluation getFeatureSelectionMetric() {
		return fsMetric;
	}
	
	public void setBiasToUniformClass(double biasToUniformClass) {
		if (biasToUniformClass<0 || biasToUniformClass>1)
			throw new IllegalArgumentException("Bias to uniform class was set to should be in [0,1], but it was set to "+biasToUniformClass);
		this.biasToUniformClass = biasToUniformClass;
	}


}
