// Java program to demonstrate working of Robot
// class. This program is for Windows. It opens
// notepad and types a message.
import java.awt.AWTException;
import java.awt.Robot;
import java.awt.event.KeyEvent;
import java.io.*;
import java.awt.event.*;
 
public class Task
{
    public static void main(String[] args) throws IOException,
                           AWTException, InterruptedException
    {
        String line;
		String pidInfo ="";

		Process p =Runtime.getRuntime().exec(System.getenv("windir") +"\\system32\\"+"tasklist.exe");

		BufferedReader input =  new BufferedReader(new InputStreamReader(p.getInputStream()));

		while ((line = input.readLine()) != null) {
			pidInfo+=line; 
		}

		input.close();

		if(pidInfo.contains("zkemnetman.exe"))
		{
			System.out.println("Found");
		}
    }
}