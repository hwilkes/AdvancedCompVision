package b6;

/*
 * Code taken from the 2014/15 COMP3204 coursework (Henry Wilkes, Alex Lay)
 * */

import java.util.HashSet;
import java.util.Set;

import org.openimaj.feature.ByteFV;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.local.engine.BasicGridSIFTEngine;
import org.openimaj.image.feature.local.keypoints.Keypoint;


public class DenseSIFTExtractor{

	public Set<ByteFV> getVectors(Set<FImage> images) {
		Set<ByteFV> vectors = new HashSet<ByteFV>();
		for(FImage f : images)
		{
			BasicGridSIFTEngine engine = new BasicGridSIFTEngine(false);

			int sifted = 0;
			LocalFeatureList<Keypoint> featurePoints = engine.findFeatures(f);
			
			for(Keypoint point : featurePoints){

				//build a sift descriptor, add to the list of sift descriptors
				vectors.add(point.getFeatureVector());
				sifted++;
				if(sifted%100 == 0){
					System.out.println(sifted + " images sifted");
				}
			}
		}


		//System.out.println("Patching complete");
		System.out.println(vectors.size() + " vectors created");

		int k = 500;
		//figure out the k means
		Set<ByteFV> vocabulary = new KMeansByteFV().getMeans(k, vectors);
		return vocabulary;
	}

}
