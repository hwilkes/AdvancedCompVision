package b6;

import java.awt.event.KeyEvent;
import java.io.File;
import java.io.IOException;

import javax.swing.JFrame;
import javax.swing.SwingUtilities;

import org.opencv.core.Core;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.feature.local.matcher.BasicMatcher;
import org.openimaj.feature.local.matcher.FastBasicKeypointMatcher;
import org.openimaj.feature.local.matcher.LocalFeatureMatcher;
import org.openimaj.feature.local.matcher.MatchingUtilities;
import org.openimaj.feature.local.matcher.consistent.ConsistentLocalFeatureMatcher2d;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.colour.RGBColour;
import org.openimaj.image.feature.local.engine.DoGSIFTEngine;
import org.openimaj.image.feature.local.engine.Engine;
import org.openimaj.image.feature.local.interest.InterestPointVisualiser;
import org.openimaj.image.feature.local.keypoints.Keypoint;
import org.openimaj.image.processing.convolution.LaplacianOfGaussian2D;
import org.openimaj.math.geometry.transforms.estimation.RobustAffineTransformEstimator;
import org.openimaj.math.model.fit.RANSAC;
import org.openimaj.video.VideoDisplay;
import org.openimaj.video.VideoDisplayListener;
import org.openimaj.video.capture.VideoCapture;
import org.openimaj.video.capture.VideoCaptureException;
import org.openimaj.video.xuggle.XuggleVideoWriter;

import com.jogamp.newt.event.KeyAdapter;

public class VideoRecorder implements VideoDisplayListener<MBFImage> {
	
	private VideoCapture video;
	private VideoDisplay displaySIFT;
	private VideoDisplay displaySURF;
	private XuggleVideoWriter writer;
    private boolean close = false;
    
    private Engine<Keypoint, FImage> engine;
    
    private MBFImage faceimage;
    private LocalFeatureList<Keypoint> facekeypoints;
    
    private JFrame matchWindow;

	
	public static void main(String args[]){
		try {
			System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
			VideoRecorder capture = new VideoRecorder(new SURFEngine());
		} catch (VideoCaptureException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public VideoRecorder(Engine<Keypoint, FImage> engine) throws IOException {
		
		this.engine = engine;
		faceimage = ImageUtilities.readMBF(new File("images/face.jpg"));
		facekeypoints = engine.findFeatures(faceimage.flatten());
		matchWindow = DisplayUtilities.createNamedWindow("MatchWindow");
		matchWindow.setVisible(true);
        //open webcam
        video = new VideoCapture(320, 240);

        //open display
        displaySIFT = VideoDisplay.createVideoDisplay(video);
        //displaySURF = VideoDisplay.createVideoDisplay(video);

        //open a writer
        writer = new XuggleVideoWriter("video.flv", video.getWidth(), video.getHeight(), 30);
        
        //set this class to listen to video display events
        displaySIFT.addVideoListener(this);

    }
	
	@Override
    public void afterUpdate(VideoDisplay<MBFImage> display) {
        //Do nothing
		
    }

    @Override
    public void beforeUpdate(MBFImage frame) {
        //write a frame 
    	
//    	LocalFeatureMatcher<Keypoint> matcher = new BasicMatcher<Keypoint>(80);
//    	matcher.setModelFeatures();
//    	matcher.findMatches(facekeypoints);
    	LocalFeatureMatcher<Keypoint> matcher = new BasicMatcher<Keypoint>(80);
    		RobustAffineTransformEstimator modelFitter = new RobustAffineTransformEstimator(5.0, 1500,
				  new RANSAC.PercentageInliersStoppingCondition(0.5));
				matcher = new ConsistentLocalFeatureMatcher2d<Keypoint>(
				  new FastBasicKeypointMatcher<Keypoint>(8), modelFitter);
		
				matcher.setModelFeatures(engine.findFeatures(frame.flatten()));
				matcher.findMatches(facekeypoints);
		
				MBFImage consistentMatches = MatchingUtilities.drawMatches(frame, faceimage, matcher.getMatches(), 
				  RGBColour.RED);
    		
    	//MBFImage basicMatches = MatchingUtilities.drawMatches(frame, faceimage, matcher.getMatches(), RGBColour.RED);
    	DisplayUtilities.display(consistentMatches, matchWindow);
    	
    	
        if (!close) {
            writer.addFrame(frame);
        }
    }

}
