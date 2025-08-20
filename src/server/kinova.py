#! /usr/bin/env python3

###
# KINOVA (R) KORTEX (TM)
#
# Copyright (c) 2018 Kinova inc. All rights reserved.
#
# This software may be modified and distributed
# under the terms of the BSD 3-Clause license.
#
# Refer to the LICENSE file for details.
#
###

import sys
import os
import time
import threading

from server import get_command, send_observation

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient

from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2, Common_pb2

# Maximum allowed waiting time during actions (in seconds)
TIMEOUT_DURATION = 20
CONTROL_DELTATIME=0.05
# Create closure to set an event after an END or an ABORT
def check_for_end_or_abort(e):
    """Return a closure checking for END or ABORT notifications

    Arguments:
    e -- event to signal when the action is completed
        (will be set when an END or ABORT occurs)
    """
    def check(notification, e = e):
        print("EVENT : " + \
              Base_pb2.ActionEvent.Name(notification.action_event))
        if notification.action_event == Base_pb2.ACTION_END \
        or notification.action_event == Base_pb2.ACTION_ABORT:
            e.set()
    return check

def example_gripper_command(base, base_cyclic, gripper_finger_delta = None):
    def get_gripper_position():
        feedback = base_cyclic.RefreshFeedback()
        return feedback.interconnect.gripper_feedback.motor[0].position
    prev_val = get_gripper_position()
    if gripper_finger_delta is None:
        gripper_finger = 0
    else:
        gripper_finger = prev_val + gripper_finger_delta*100
        gripper_finger = min(100, max(0, gripper_finger)) # clamp gripper finger
    # send gripper command from gripper_finger_delta (actual action)
    gripper_cmd = Base_pb2.GripperCommand()
    finger = gripper_cmd.gripper.finger.add()
    gripper_cmd.mode = Base_pb2.GRIPPER_POSITION
    finger.finger_identifier = 1
    finger.value = gripper_finger / 100
    if abs(gripper_finger - prev_val) > 0.001:
        base.SendGripperCommand(gripper_cmd)
        # gripper_finger_delta will be None when resetting gripper
        time.sleep(2 if gripper_finger_delta is None else CONTROL_DELTATIME)
        base.Stop()
        time.sleep(0.01)
    # Get new gripper position
    gripper_pos1 = get_gripper_position()
    # Send new gripper command to try and close the gripper a small amount
    gripper_cmd = Base_pb2.GripperCommand()
    finger = gripper_cmd.gripper.finger.add()
    gripper_cmd.mode = Base_pb2.GRIPPER_POSITION
    finger.finger_identifier = 1
    finger.value = min(1.0, max(0.0, gripper_pos1/100 + 0.001))
    base.SendGripperCommand(gripper_cmd)
    time.sleep(0.1)
    # Get new gripper position
    gripper_pos2 = get_gripper_position()
    # Compare them to see if the gripper actually was able to close
    if abs(gripper_pos1 - gripper_pos2) >= 0.0008:
        return 0 # If they have changed then not gripping 
    return 1 # If they not changed then gripping 

def example_twist_command(base, end_effector_delta):
    command = Base_pb2.TwistCommand()

    command.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_TOOL
    command.duration = 0

    twist = command.twist
    twist.linear_x = end_effector_delta[0]
    twist.linear_y = end_effector_delta[1]
    twist.linear_z = end_effector_delta[2]
    twist.angular_x = 0
    twist.angular_y = 0
    twist.angular_z = 0

    print ("Sending the twist command for 5 seconds...")
    base.SendTwistCommand(command)

    # Let time for twist to be executed
    time.sleep(CONTROL_DELTATIME)

    print ("Stopping the robot...")
    base.Stop()
    time.sleep(0.01)

def example_cartesian_action_movement(base, base_cyclic, end_effector_delta):
    feedback = base_cyclic.RefreshFeedback()

    is_gripped = example_gripper_command(base, base_cyclic, end_effector_delta[3])
    
    print("Starting Cartesian action movement ...")
    action = Base_pb2.Action()
    action.name = "Example Cartesian action movement"
    action.application_data = ""

    cartesian_pose = action.reach_pose.target_pose
    cartesian_pose.x = feedback.base.tool_pose_x + end_effector_delta[0]        # (meters)
    cartesian_pose.y = feedback.base.tool_pose_y + end_effector_delta[1]     # (meters)
    cartesian_pose.z = feedback.base.tool_pose_z + end_effector_delta[2]     # (meters)
    cartesian_pose.theta_x = 0 # (degrees)
    cartesian_pose.theta_y = feedback.base.tool_pose_theta_y # (degrees)
    cartesian_pose.theta_z = feedback.base.tool_pose_theta_z # (degrees)

    print("Z position", feedback.base.tool_pose_z )
    if feedback.base.tool_pose_z < 0.05:
        # action.reach_pose.target_pose.z = 0 # don't move
        end_effector_delta[2] = min(0, end_effector_delta[2])

    e = threading.Event()
    notification_handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )

    print("Executing action")
        
    if end_effector_delta[3] is None:
        base.ExecuteAction(action)
    else:
        example_twist_command(base, end_effector_delta)
    feedback = base_cyclic.RefreshFeedback()

    print("Waiting for movement to finish ...")
    finished = e.wait(TIMEOUT_DURATION)
    base.Unsubscribe(notification_handle)

    if finished:
        print("Cartesian movement completed")
    else:
        print("Timeout on action notification wait")

    print("getting observations")
    feedback = base_cyclic.RefreshFeedback()

    cartesian_pose = action.reach_pose.target_pose
    cartesian_pose.x = feedback.base.tool_pose_x   # (meters)
    cartesian_pose.y = feedback.base.tool_pose_y   # (meters)
    cartesian_pose.z = feedback.base.tool_pose_z   # (meters)
    cartesian_pose.theta_x = feedback.base.tool_pose_theta_x # (degrees)
    cartesian_pose.theta_y = feedback.base.tool_pose_theta_y # (degrees)
    cartesian_pose.theta_z = feedback.base.tool_pose_theta_z # (degrees)
    obs = [cartesian_pose.x, cartesian_pose.y, cartesian_pose.z, feedback.interconnect.gripper_feedback.motor[0].position / 100, is_gripped]

    return finished, obs

def example_angular_action_movement(base, joint_angles):
    
    print("Starting angular action movement ...")
    action = Base_pb2.Action()
    action.name = "Example angular action movement"
    action.application_data = ""

    actuator_count = base.GetActuatorCount()

    assert actuator_count.count == len(joint_angles), "Incorrect number of joint angles passed, should be " +  str(actuator_count.count)

    # Place arm straight up
    for joint_id in range(actuator_count.count):
        joint_angle = action.reach_joint_angles.joint_angles.joint_angles.add()
        joint_angle.joint_identifier = joint_id
        joint_angle.value = joint_angles[joint_id]

    e = threading.Event()
    notification_handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )
    
    print("Executing action")
    base.ExecuteAction(action)

    print("Waiting for movement to finish ...")
    finished = e.wait(TIMEOUT_DURATION)
    base.Unsubscribe(notification_handle)

    if finished:
        print("Angular movement completed")
    else:
        print("Timeout on action notification wait")
    return finished

def example_move_to_home_position(base, base_cyclic):
    # Move arm to ready position
    print("Moving the arm to a safe position")
    finished = example_angular_action_movement(base, [
        39.88359211905661,
        63.76447302010932,
        100.11864512116006,
        229.26822,
        74.9657979149169,
        294.172879,
        270.0006313774491
    ])
    # TODO: Ensure that gripper starts at some fixed pose
    if finished:
        print("Safe position reached")
        finished, _ = example_cartesian_action_movement(base, base_cyclic, [0, 0, -0.03, None])
        if finished:
            print("Start position reached")
        else:
            print("Timeout on action notification wait")
    else:
        print("Timeout on action notification wait")
    return finished

def old_example_move_to_home_position(base):
    # Make sure the arm is in Single Level Servoing mode
    base_servo_mode = Base_pb2.ServoingModeInformation()
    base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
    base.SetServoingMode(base_servo_mode)
    
    # Move arm to ready position
    print("Moving the arm to a safe position")
    action_type = Base_pb2.RequestedActionType()
    action_type.action_type = Base_pb2.REACH_JOINT_ANGLES
    action_list = base.ReadAllActions(action_type)
    action_handle = None
    for action in action_list.action_list:
        if action.name == "Home":
            action_handle = action.handle

    if action_handle == None:
        print("Can't reach safe position. Exiting")
        return False

    e = threading.Event()
    notification_handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )

    base.ExecuteActionFromReference(action_handle)
    finished = e.wait(TIMEOUT_DURATION)
    base.Unsubscribe(notification_handle)

    if finished:
        print("Safe position reached")
    else:
        print("Timeout on action notification wait")
    return finished

def main():
    
    # Import the utilities helper module
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))
    import utilities

    # Parse arguments
    args = utilities.parseConnectionArguments()
    
    # Create connection to the device and get the router
    with utilities.DeviceConnection.createTcpConnection(args) as router:
        # Create required services
        base = BaseClient(router)
        base_cyclic = BaseCyclicClient(router)

        # Example core
        success = True
        
        success_status = example_move_to_home_position(base, base_cyclic)
        success &= success_status

        while True:
            end_eff_translation, restart = get_command() # blocking socket communication -- temporary
            print("action before",end_eff_translation)
            if restart:
                example_move_to_home_position(base, base_cyclic)
            else:
                success_status, observations = example_cartesian_action_movement(base, base_cyclic, end_eff_translation)
                success &= success_status
                send_observation(observations)
                

            # You can also refer to the 110-Waypoints examples if you want to execute
            # a trajectory defined by a series of waypoints in joint space or in Cartesian space

            # return 0 if success else 1

if __name__ == "__main__":
    exit(main())