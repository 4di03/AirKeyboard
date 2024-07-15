#include <iostream>
#include <cstdlib>
#include <cstring>
#include <ApplicationServices/ApplicationServices.h>
using namespace std;


CGPoint getMouseLocation() {
    CGEventRef event = CGEventCreate(NULL);
    CGPoint cursor = CGEventGetLocation(event);
    CFRelease(event);
    return cursor;
}

class MouseController{
    private:
        CGPoint mousePoint;

    public:
        MouseController()
        {

        }

        CGPoint getMouseLocation(){
            return  getMouseLocation();
        }

        void setMouseLocation(CGPoint mousePoint){
                        // Move the mouse
            CGEventRef moveEvent = CGEventCreateMouseEvent(
                NULL, kCGEventMouseMoved,
                mousePoint, kCGHIDEventTap
            );
            CGEventPost(kCGHIDEventTap, moveEvent);
            CFRelease(moveEvent);
            return;
        }
        
        void moveMouse(int deltaX,int deltaY){
            // Set the new mouse location
            CGPoint mousePoint = this->getMouseLocation();
            mousePoint.x += deltaX;
            mousePoint.y += deltaY;
            this->setMouseLocation(mousePoint);

        }

        void setMouse(int deltaX,int deltaY){
            // Set the new mouse location
            CGPoint mousePoint; 
            mousePoint.x = deltaX;
            mousePoint.y = deltaY;

            // Move the mouse
            this->setMouseLocation(mousePoint);

        }

        void setMouseToCurrentLocation(){
            // Set the new mouse location
            CGPoint mousePoint = this->getMouseLocation();

            // Move the mouse
            this->setMouseLocation(mousePoint);

        }
        

        void leftClick(){

        // Simulate a left mouse button click
        CGEventRef leftClickEvent = CGEventCreateMouseEvent(
            NULL, kCGEventLeftMouseDown,
            mousePoint, kCGHIDEventTap
        );
        CGEventPost(kCGHIDEventTap, leftClickEvent);
        CFRelease(leftClickEvent);

        // Release the left mouse button
        CGEventRef leftReleaseEvent = CGEventCreateMouseEvent(
            NULL, kCGEventLeftMouseUp,
            mousePoint, kCGHIDEventTap
        );
        CGEventPost(kCGHIDEventTap, leftReleaseEvent);
        CFRelease(leftReleaseEvent);
        return;

        }
};

int main() {

    // Get the current mouse location
    int ct;
    while (true){
    MouseController mc = MouseController();

    if (ct %1000000000 == 0){
    // Set the new mouse location
    mc.setMouseToCurrentLocation();
    cout << "Clicking" << endl;
    mc.leftClick();
    ct = 0;
    }
    ct++;

    } 

    return 0;
}
