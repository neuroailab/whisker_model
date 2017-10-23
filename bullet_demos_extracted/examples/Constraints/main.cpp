/*
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2015 Google Inc. http://bulletphysics.org

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/


#include "TestHingeTorque.h"

#include "../CommonInterfaces/CommonExampleInterface.h"
#include "../CommonInterfaces/CommonGUIHelperInterface.h"
#include "BulletCollision/CollisionDispatch/btCollisionObject.h"
#include "BulletCollision/CollisionShapes/btCollisionShape.h"
#include "BulletDynamics/Dynamics/btDiscreteDynamicsWorld.h"


#include "LinearMath/btTransform.h"
#include "LinearMath/btHashMap.h"

#include <iostream>


int main(int argc, char* argv[])
{
	
    //std::cout << "Test output" << std::endl;
    //std::cout << argv[1] << std::endl;
	DummyGUIHelper noGfx;

	CommonExampleOptions options(&noGfx);
    //options.m_guiHelper->setconfigname("/Users/chengxuz/barrel/bullet/barrel_github/cmd_gen_mp4/test.cfg");
    options.m_guiHelper->setconfigname(argv[1]);
    options.m_guiHelper->have_visualize = 100;
	CommonExampleInterface*    example = TestHingeTorqueCreateFunc(options);
	example->initPhysics();
    while (true){
        example->stepSimulation(1.f/240.f);
    }
	example->exitPhysics();

	delete example;
    std::cout << "Test output finished!" << std::endl;

	return 0;
}
