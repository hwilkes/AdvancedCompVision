package b6;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListBackedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.MapBackedDataset;
import org.openimaj.experiment.dataset.sampling.GroupSampler;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.feature.ArrayFeatureVector;
import org.openimaj.feature.ByteFV;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.dense.gradient.dsift.ByteDSIFTKeypoint;
import org.openimaj.image.feature.dense.gradient.dsift.DenseSIFT;
import org.openimaj.image.feature.dense.gradient.dsift.PyramidDenseSIFT;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.BlockSpatialAggregator;
import org.openimaj.image.feature.local.aggregate.VectorAggregator;
import org.openimaj.image.feature.local.engine.DoGSIFTEngine;
import org.openimaj.image.feature.local.keypoints.Keypoint;
import org.openimaj.ml.annotation.ScoredAnnotation;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator.Mode;
import org.openimaj.ml.clustering.ByteCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.ByteKMeans;
import org.openimaj.util.pair.IntFloatPair;

import de.bwaldvogel.liblinear.SolverType;


public class SIFTAndSURF {
	DoGSIFTEngine siftengine;
	SURFEngine surfengine;
	private String[] classes;
	private boolean trained = false;
	private ByteCentroidsResult clusters;
	
	private LiblinearAnnotator<FImage, String> siftann;
	private LiblinearAnnotator<FImage, String> surfann;
	
	
	public SIFTAndSURF(){
		siftengine = new DoGSIFTEngine();	
		surfengine = new SURFEngine();	
	}
	/*
	 * Makes some use of code snippets from http://www.openimaj.org/tutorial/classification101.html
	 * */
	public void trainImages(String[] classes, Map<String,FImage[]> images){
		this.classes = classes;
		GroupedDataset<String,ListDataset<FImage>,FImage> dataset = new MapBackedDataset<String,ListDataset<FImage>,FImage>();
		
		for(Entry<String,FImage[]> e : images.entrySet()){
			ListDataset<FImage> listdata = new ListBackedDataset<FImage>();
			for(FImage f : e.getValue()){
				listdata.add(f);
			}	
			dataset.put(e.getKey(), listdata);
		}
		System.out.println("Grouped Dataset created");
		Set<ByteFV> vectorssift = new HashSet<ByteFV>();
		Set<ByteFV> vectorssurf = new HashSet<ByteFV>();
		
		List<LocalFeatureList<Keypoint>> featurevectorsone = new ArrayList<LocalFeatureList<Keypoint>>();
		List<LocalFeatureList<Keypoint>> featurevectorstwo = new ArrayList<LocalFeatureList<Keypoint>>();
		//DenseSIFT sifter = new DenseSIFT(16,16);
		
		for(Entry<String,FImage[]> e : images.entrySet()){
			System.out.println("Training class: "+e.getKey());
			int num = 0;
			for(FImage f : e.getValue()){
				//engine.analyseImage(f);
				//LocalFeatureList<ByteDSIFTKeypoint> featurePoints = sifter.getByteKeypoints();
				LocalFeatureList<Keypoint> featurePoints1 = siftengine.findFeatures(f);
				featurevectorsone.add(featurePoints1);
				for(Keypoint point : featurePoints1){
					vectorssift.add(point.getFeatureVector());
				}
				
				LocalFeatureList<Keypoint> featurePoints2 = siftengine.findFeatures(f);
				featurevectorstwo.add(featurePoints2);
				for(Keypoint point : featurePoints2){
					vectorssurf.add(point.getFeatureVector());
				}
//				num++;
//				if(num == 3){
//					break;
//				}
			}
		}
		
		int k = 500;
		
		KMeansByteFV kmeans = new KMeansByteFV();
		Set<ByteFV> vocabulary = new KMeansByteFV().getMeans(k, vectorssift);
		kmeans.getMeans(k, vocabulary);
		
		//DenseSIFT dsift = new DenseSIFT(5, 7);
		//PyramidDenseSIFT<FImage> pdsift = new PyramidDenseSIFT<FImage>(dsift, 6f, 7);
		ByteFV[] array = new ByteFV[vocabulary.size()];
		FeatureExtractor<SparseIntFV, FImage> extractor = new BOVWExtractorByte(Arrays.asList(vocabulary.toArray(array)),siftengine);

		siftann = new LiblinearAnnotator<FImage, String>(
	            extractor, Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
		System.out.println("Training liblinearannotator for sft");
		siftann.train(dataset);
		
		KMeansByteFV kmeanssurf = new KMeansByteFV();
		Set<ByteFV> vocabularysurf = new KMeansByteFV().getMeans(k, vectorssurf);
		kmeanssurf.getMeans(k, vocabulary);
		
		//DenseSIFT dsift = new DenseSIFT(5, 7);
		//PyramidDenseSIFT<FImage> pdsift = new PyramidDenseSIFT<FImage>(dsift, 6f, 7);
		ByteFV[] arraysurf = new ByteFV[vocabularysurf.size()];
		FeatureExtractor<SparseIntFV, FImage> extractorsurf = new BOVWExtractorByte(Arrays.asList(vocabularysurf.toArray(arraysurf)),surfengine);

		siftann = new LiblinearAnnotator<FImage, String>(
				extractorsurf, Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
		System.out.println("Training liblinearannotator for surf");
		siftann.train(dataset);
		trained = true;
	}
	
	public List<ScoredAnnotation<String>> classifySIFT(FImage image){
		if(trained){			
			return siftann.annotate(image);
					}
		else{
			return null;
		}
	}
	
	public List<ScoredAnnotation<String>> classifySURF(FImage image){
		if(trained){			
			return siftann.annotate(image);
					}
		else{
			return null;
		}
	}
}
