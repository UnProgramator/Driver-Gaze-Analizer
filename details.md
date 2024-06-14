The [imageReader package](https://github.com/UnProgramator/Driver-Gaze-Analizer/tree/master/DGA/imageReaders) contains the classes for reading files. 
The [DrivfaceInput class](https://github.com/UnProgramator/Driver-Gaze-Analizer/blob/master/DGA/imageReaders/DrivfaceInput.py) contains an example of reading the files for the DriveFace dataset {[https://archive.ics.uci.edu/dataset/378/drivface](https://archive.ics.uci.edu/dataset/378/drivface)}

The [drvalidation function](https://github.com/UnProgramator/Driver-Gaze-Analizer/blob/SACI/DGA/utilities/Validation/dreyeve_validation.py) is an example of testing the word encoding and clustering eficiency. This method bypass the gaze estimation using the L2CS-NET!

The base interface [ImageReader](https://github.com/UnProgramator/Driver-Gaze-Analizer/blob/master/DGA/imageReaders/ImageReader.py) contains the methoods which need to be implemented for the program to work as intended, to be more specific the two methods defined in the interface are called by the [Processor class](https://github.com/UnProgramator/Driver-Gaze-Analizer/blob/master/DGA/backbone/processor.py). The prototiyes also contain the expected return type.

In the [main.py file](https://github.com/UnProgramator/Driver-Gaze-Analizer/blob/master/DGA/main.py) are 4 function (f1 to f4) with examples of howw to call the Processor for different operations.
* f1 - testing the codate_aparition method
* f2 - testing two words encoding for the action words, to be given then to the clustering algorithm. Used to obatin the results for the paper
* f3 - testing action list
* f4 - a more elegant way to display the output of the validation method, used to analyze the results

To call the function, you will have to create a ImageReader subobject (for example a DrivfaceInput instance)  and give it to the processor class, as follows
`imgsrc = DrivfaceInput((1,2,3))
f4(imgsrc)`

Other usefull functions in the Processor class:
the `validate(self, imInput) -> list[tuple[str, str,str,str]]` [method](https://github.com/UnProgramator/Driver-Gaze-Analizer/blob/master/DGA/backbone/processor.py#L265) 
--outputs the direction of gaze for each frame, together with the frame name. For a more thorough verification, I recomend to output to a file, by redirecting the standard output

the `render(self,imInput:ImageReader, savePath:str=None)` [method](https://github.com/UnProgramator/Driver-Gaze-Analizer/blob/master/DGA/backbone/processor.py#L232)
-- usefull for a visual verification or validation. It needs an ImageReader for obtaining the pictures to render. If savePath is not `None`, then it saves the pictures in the given directory
