// Java program to demonstrate working of Robot
// class. This program is for Windows. It opens
// notepad and types a message.
import java.awt.AWTException;
import java.awt.Robot;
import java.awt.event.KeyEvent;
import java.io.*;
import java.awt.event.*;
 
public class robo
{
    public static void main(String[] args) throws IOException,
                           AWTException, InterruptedException
    {
		
		String line;
		String pidInfo ="";

		Process p =Runtime.getRuntime().exec(System.getenv("windir") +"\\system32\\"+"tasklist.exe");

		BufferedReader input =  new BufferedReader(new InputStreamReader(p.getInputStream()));

		while ((line = input.readLine()) != null) {
			//System.out.println(line);
			pidInfo+=line; 
		}

		input.close();
		
		if(pidInfo.contains("zkemnetman.exe"))
		{
			System.out.println("Found");
		}else{
			String command = "C:/Program Files (x86)/ZKTeco/zkemnetman/zkemnetman.exe";
			Runtime run = Runtime.getRuntime();
			run.exec(command);
		}
        try {
            Thread.sleep(2000);
        }
        catch (InterruptedException e)
        {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
 
        // Create an instance of Robot class
        Robot robot = new Robot();
 
        // Press keys using robot. A gap of
        // of 500 milli seconds is added after
        // every key press
        // robot.keyPress(KeyEvent.VK_H);
        Thread.sleep(50);
		robot.mouseMove(50,135);
		// press the mouse
		Thread.sleep(50);
        robot.mousePress(InputEvent.BUTTON1_MASK);
		//Thread.sleep(50);
		robot.mouseRelease(InputEvent.BUTTON1_MASK);
		Thread.sleep(50);
		robot.mouseMove(100,280);
		Thread.sleep(50);
		robot.mousePress(InputEvent.BUTTON1_MASK);
		//Thread.sleep(50);
		robot.mouseRelease(InputEvent.BUTTON1_MASK);
		Thread.sleep(50);
		// press the alt key
		robot.keyPress(KeyEvent.VK_ALT);
		//Thread.sleep(50);
		robot.keyRelease(KeyEvent.VK_ALT);
		Thread.sleep(50);
		// go to right (->)
		robot.keyPress(KeyEvent.VK_RIGHT);
		//Thread.sleep(50);
		robot.keyRelease(KeyEvent.VK_RIGHT);
		Thread.sleep(50);
		robot.keyPress(KeyEvent.VK_RIGHT);
		//Thread.sleep(50);
		robot.keyRelease(KeyEvent.VK_RIGHT);
		Thread.sleep(50);
		// press the down arrow
		robot.keyPress(KeyEvent.VK_DOWN);
		robot.keyRelease(KeyEvent.VK_DOWN);
		Thread.sleep(50);
		robot.keyPress(KeyEvent.VK_DOWN);
		robot.keyRelease(KeyEvent.VK_DOWN);
		Thread.sleep(50);
		// press enter
		robot.keyPress(KeyEvent.VK_ENTER);
		Thread.sleep(50);
    }
}