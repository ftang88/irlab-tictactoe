# Copyright (c) 2023 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

import argparse
import math
import sys
import time

import cv2
import numpy as np
from google.protobuf import wrappers_pb2

import bosdyn.client
import bosdyn.client.util
from bosdyn.api import (basic_command_pb2, geometry_pb2, image_pb2, manipulation_api_pb2,
                        network_compute_bridge_pb2, robot_state_pb2)
from bosdyn.client import frame_helpers, math_helpers
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.network_compute_bridge_client import NetworkComputeBridgeClient
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient,
                                         block_for_trajectory_cmd, block_until_arm_arrives)
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.world_object import WorldObjectClient
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive

from dotenv import load_dotenv

import os
import datetime

kImageSources = [
    'frontleft_fisheye_image', 'frontright_fisheye_image', 'left_fisheye_image',
    'right_fisheye_image', 'back_fisheye_image'
]

# save_dir = "test_dataset"
# os.makedirs(save_dir, exist_ok=True)

def get_obj_and_img(network_compute_client, server, model, confidence, image_sources, label):

    for source in image_sources:
        # Build a network compute request for this image source.
        image_source_and_service = network_compute_bridge_pb2.ImageSourceAndService(
            image_source=source)

        # Input data:
        #   model name
        #   minimum confidence (between 0 and 1)
        #   if we should automatically rotate the image
        input_data = network_compute_bridge_pb2.NetworkComputeInputData(
            image_source_and_service=image_source_and_service, model_name=model,
            min_confidence=confidence, rotate_image=network_compute_bridge_pb2.
            NetworkComputeInputData.ROTATE_IMAGE_ALIGN_HORIZONTAL)

        # Server data: the service name
        server_data = network_compute_bridge_pb2.NetworkComputeServerConfiguration(
            service_name=server)

        # Pack and send the request.
        process_img_req = network_compute_bridge_pb2.NetworkComputeRequest(
            input_data=input_data, server_config=server_data)
        
        resp = network_compute_client.network_compute_bridge_command(process_img_req)

        # for attempt in range(2):
        #     try:
        #         resp = network_compute_client.network_compute_bridge_command(process_img_req)
        #         if resp.status != network_compute_bridge_pb2.NetworkComputeStatus.STATUS_UNKNOWN:
        #             break
        #         print(f"Retry {attempt +1}] got UUNKNOWN status")
        #     except Exception as e:
        #         print(f"Retry {attempt + 1} Exception during network computer: {e}")
            
        #     time.sleep(0.5)
        # else:
        #     raise RuntimeError("Detection failed after 2 attempts")

        best_obj = None
        highest_conf = 0.0
        best_vision_tform_obj = None

        img = get_bounding_box_image(resp)
        image_full = resp.image_response

        # # save img for testing
        # filename = os.path.join(save_dir, f"{source}_{datetime.datetime.now():%Y%m%d_%H%M%S}.png")
        # cv2.imwrite(filename, img)
        # print(f"Saved {filename}")

        # Show the image
        cv2.imshow('X-piece detection', img)
        cv2.waitKey(1)

        if len(resp.object_in_image) > 0:
            for obj in resp.object_in_image:
                # Get the label
                obj_label = obj.name.split('_label_')[-1]
                if obj_label != label:
                    continue
                conf_msg = wrappers_pb2.FloatValue()
                obj.additional_properties.Unpack(conf_msg)
                conf = conf_msg.value

                try:
                    vision_tform_obj = frame_helpers.get_a_tform_b(
                        obj.transforms_snapshot, frame_helpers.VISION_FRAME_NAME,
                        obj.image_properties.frame_name_image_coordinates)
                except bosdyn.client.frame_helpers.ValidateFrameTreeError:
                    # No depth data available.
                    vision_tform_obj = None

                if conf > highest_conf and vision_tform_obj is not None:
                    highest_conf = conf
                    best_obj = obj
                    best_vision_tform_obj = vision_tform_obj

        if best_obj is not None:
            return best_obj, image_full, best_vision_tform_obj

    return None, None, None


def get_bounding_box_image(response):
    dtype = np.uint8
    img = np.fromstring(response.image_response.shot.image.data, dtype=dtype)
    if response.image_response.shot.image.format == image_pb2.Image.FORMAT_RAW:
        img = img.reshape(response.image_response.shot.image.rows,
                          response.image_response.shot.image.cols)
    else:
        img = cv2.imdecode(img, -1)

    # Convert to BGR so we can draw colors
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Draw bounding boxes in the image for all the detections.
    for obj in response.object_in_image:
        conf_msg = wrappers_pb2.FloatValue()
        obj.additional_properties.Unpack(conf_msg)
        confidence = conf_msg.value

        polygon = []
        min_x = float('inf')
        min_y = float('inf')
        for v in obj.image_properties.coordinates.vertexes:
            polygon.append([v.x, v.y])
            min_x = min(min_x, v.x)
            min_y = min(min_y, v.y)

        polygon = np.array(polygon, np.int32)
        polygon = polygon.reshape((-1, 1, 2))
        cv2.polylines(img, [polygon], True, (0, 255, 0), 2)

        caption = "{} {:.3f}".format(obj.name, confidence)
        cv2.putText(img, caption, (int(min_x), int(min_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)

    return img


def find_center_px(polygon):
    min_x = math.inf
    min_y = math.inf
    max_x = -math.inf
    max_y = -math.inf
    for vert in polygon.vertexes:
        if vert.x < min_x:
            min_x = vert.x
        if vert.y < min_y:
            min_y = vert.y
        if vert.x > max_x:
            max_x = vert.x
        if vert.y > max_y:
            max_y = vert.y
    x = math.fabs(max_x - min_x) / 2.0 + min_x
    y = math.fabs(max_y - min_y) / 2.0 + min_y
    return (x, y)


PICK_UP_STATE_FAIL = 0
PICK_UP_STATE_SUCCESS = 1
PICK_UP_STATE_NOT_FOUND = 2
MAX_TRY_COUNT = 3

def pick_up(options, robot):

    # Time sync is necessary so that time-based filter requests can be converted
    robot.time_sync.wait_for_sync()

    network_compute_client = robot.ensure_client(NetworkComputeBridgeClient.default_service_name)
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    command_client = robot.ensure_client(RobotCommandClient.default_service_name)
    manipulation_api_client = robot.ensure_client(ManipulationApiClient.default_service_name)
    # This script assumes the robot is already standing via the tablet.  We'll take over from the
    # tablet.
    _model = options.model
    _ml_service = options.ml_service
    confidence = options.confidence_piece
    
    vision_tform_hand_at_drop = None

    try_count = 0

    while True:
        holding_piece = False
        while not holding_piece:

            # Capture an image and run ML on it.
            X, image, vision_tform_dogtoy = get_obj_and_img(
                network_compute_client, _ml_service, _model,
                confidence, kImageSources, 'X')

            if try_count == MAX_TRY_COUNT:
                return PICK_UP_STATE_NOT_FOUND

            if X is None:
                # Didn't find anything, keep searching.
                print("Didn't find anything")
                try_count += 1
                continue

            # If we have already dropped the X Piece off, make sure it has moved a sufficient amount before
            # picking it up again
            # if vision_tform_hand_at_drop is not None and pose_dist(
            #         vision_tform_hand_at_drop, vision_tform_dogtoy) < 0.5:
            #     print('Found X, but it hasn\'t moved.  Waiting...')
            #     time.sleep(1)
            #     continue

            print('Found X...')

            # Got a X.  Request pick up.

            # Stow the arm in case it is deployed
            stow_cmd = RobotCommandBuilder.arm_stow_command()
            command_client.robot_command(stow_cmd)

            # NOTE: we'll enable this code in Part 5, when we understand it.
            # -------------------------
            # Walk to the object.
            walk_rt_vision, heading_rt_vision = compute_stand_location_and_yaw(
                vision_tform_dogtoy, robot_state_client, distance_margin=0.8)

            se2_pose = geometry_pb2.SE2Pose(
                position=geometry_pb2.Vec2(x=walk_rt_vision[0], y=walk_rt_vision[1]),
                angle=heading_rt_vision)
            move_cmd = RobotCommandBuilder.synchro_se2_trajectory_command(
                se2_pose,
                frame_name=frame_helpers.VISION_FRAME_NAME,
                params=get_walking_params(0.5, 0.5))
            end_time = 5.0
            cmd_id = command_client.robot_command(command=move_cmd,
                                                    end_time_secs=time.time() +
                                                    end_time)

            # Wait until the robot reports that it is at the goal.
            block_for_trajectory_cmd(command_client, cmd_id, timeout_sec=5)
            # -------------------------

            # The ML result is a bounding box.  Find the center.
            (center_px_x, center_px_y) = find_center_px(X.image_properties.coordinates)

            # Request Pick Up on that pixel.
            pick_vec = geometry_pb2.Vec2(x=center_px_x, y=center_px_y)
            grasp = manipulation_api_pb2.PickObjectInImage(
                pixel_xy=pick_vec,
                transforms_snapshot_for_camera=image.shot.transforms_snapshot,
                frame_name_image_sensor=image.shot.frame_name_image_sensor,
                camera_model=image.source.pinhole)

            # We can specify where in the gripper we want to grasp. About halfway is generally good for
            # small objects like this. For a bigger object like a shoe, 0 is better (use the entire
            # gripper)
            grasp.grasp_params.grasp_palm_to_fingertip = 0.6

            # Tell the grasping system that we want a top-down grasp.

            # Add a constraint that requests that the x-axis of the gripper is pointing in the
            # negative-z direction in the vision frame.

            # The axis on the gripper is the x-axis.
            axis_on_gripper_ewrt_gripper = geometry_pb2.Vec3(x=1, y=0, z=0)

            # The axis in the vision frame is the negative z-axis
            axis_to_align_with_ewrt_vision = geometry_pb2.Vec3(x=0, y=0, z=-1)

            # Add the vector constraint to our proto.
            constraint = grasp.grasp_params.allowable_orientation.add()
            constraint.vector_alignment_with_tolerance.axis_on_gripper_ewrt_gripper.CopyFrom(
                axis_on_gripper_ewrt_gripper)
            constraint.vector_alignment_with_tolerance.axis_to_align_with_ewrt_frame.CopyFrom(
                axis_to_align_with_ewrt_vision)

            # We'll take anything within about 15 degrees for top-down or horizontal grasps.
            constraint.vector_alignment_with_tolerance.threshold_radians = 0.1

            # Specify the frame we're using.
            grasp.grasp_params.grasp_params_frame_name = frame_helpers.VISION_FRAME_NAME

            # Build the proto
            grasp_request = manipulation_api_pb2.ManipulationApiRequest(
                pick_object_in_image=grasp)

            # Send the request
            print('Sending grasp request...')
            cmd_response = manipulation_api_client.manipulation_api_command(
                manipulation_api_request=grasp_request)

            # Wait for the grasp to finish
            grasp_done = False
            failed = False
            time_start = time.time()
            while not grasp_done:
                feedback_request = manipulation_api_pb2.ManipulationApiFeedbackRequest(
                    manipulation_cmd_id=cmd_response.manipulation_cmd_id)

                # Send a request for feedback
                response = manipulation_api_client.manipulation_api_feedback_command(
                    manipulation_api_feedback_request=feedback_request)
                current_state = response.current_state
                current_time = time.time() - time_start
                print(
                    'Current state ({time:.1f} sec): {state}'.format(
                        time=current_time,
                        state=manipulation_api_pb2.ManipulationFeedbackState.Name(
                            current_state)), end='                \r')
                sys.stdout.flush()

                failed_states = [
                    manipulation_api_pb2.MANIP_STATE_GRASP_FAILED,
                    manipulation_api_pb2.MANIP_STATE_GRASP_PLANNING_NO_SOLUTION,
                    manipulation_api_pb2.MANIP_STATE_GRASP_FAILED_TO_RAYCAST_INTO_MAP,
                    manipulation_api_pb2.MANIP_STATE_GRASP_PLANNING_WAITING_DATA_AT_EDGE
                ]

                failed = current_state in failed_states
                grasp_done = current_state == manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED or failed

                time.sleep(0.25)

            time.sleep(0.25)
            if current_state == manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED:
                print(robot_state_client.get_robot_state().manipulator_state.gripper_open_percentage)
                gripper_degree = robot_state_client.get_robot_state().manipulator_state.gripper_open_percentage
                print("Gripper Degree Percentage:", gripper_degree)
                #checks to be sure gripper degree is not equal or less than 0
                if gripper_degree <= 10:
                    holding_piece = False
                    grasp_holding_override = manipulation_api_pb2.ApiGraspOverride(
                                override_request=manipulation_api_pb2.ApiGraspOverride.OVERRIDE_HOLDING)
                            
                    carriable_and_stowable_override = manipulation_api_pb2.ApiGraspedCarryStateOverride(
                                override_request=robot_state_pb2.ManipulatorState.CARRY_STATE_CARRIABLE_AND_STOWABLE)
                            
                    override_request = manipulation_api_pb2.ApiGraspOverrideRequest(
                                api_grasp_override=grasp_holding_override,
                                carry_state_override=carriable_and_stowable_override)
                    manipulation_api_client.grasp_override_command(override_request)

                    wait_until_grasp_state_updates(override_request, robot_state_client)
                    stow = RobotCommandBuilder.arm_stow_command()

                    block_until_arm_arrives(command_client, command_client.robot_command(stow), 3.0)
                    print("Failed to grab")
                    return PICK_UP_STATE_FAIL
                else:
                    holding_piece = not failed       
            else:
                holding_piece = not failed
                

            
        time.sleep(0.25)
        # Move the arm to a carry position.
        grasp_holding_override = manipulation_api_pb2.ApiGraspOverride(
            override_request=manipulation_api_pb2.ApiGraspOverride.OVERRIDE_HOLDING)
        
        carriable_and_stowable_override = manipulation_api_pb2.ApiGraspedCarryStateOverride(
            override_request=robot_state_pb2.ManipulatorState.CARRY_STATE_CARRIABLE_AND_STOWABLE)
        
        override_request = manipulation_api_pb2.ApiGraspOverrideRequest(
            api_grasp_override=grasp_holding_override,
            carry_state_override=carriable_and_stowable_override)
        manipulation_api_client.grasp_override_command(override_request)

        # Wait for the override to take effect before trying to move the arm.
        wait_until_grasp_state_updates(override_request, robot_state_client)

        print('')
        print('Grasp finished, Carrying...')
        carry_cmd = RobotCommandBuilder.arm_carry_command()
        
        block_until_arm_arrives(command_client, command_client.robot_command(carry_cmd), 2.0)

        # print('Carrying Finished, Stowing...')
        # stow = RobotCommandBuilder.arm_stow_command()

        # block_until_arm_arrives(command_client, command_client.robot_command(stow), 3.0)
                    
        # Wait for the stow command to finish
        time.sleep(0.25)
        return PICK_UP_STATE_SUCCESS
    

def compute_stand_location_and_yaw(vision_tform_target, robot_state_client, distance_margin):
    # Compute drop-off location:
    #   Draw a line from Spot to the person
    #   Back up 2.0 meters on that line
    vision_tform_robot = frame_helpers.get_a_tform_b(
        robot_state_client.get_robot_state().kinematic_state.transforms_snapshot,
        frame_helpers.VISION_FRAME_NAME, frame_helpers.GRAV_ALIGNED_BODY_FRAME_NAME)

    # Compute vector between robot and person
    robot_rt_person_ewrt_vision = [
        vision_tform_robot.x - vision_tform_target.x, vision_tform_robot.y - vision_tform_target.y,
        vision_tform_robot.z - vision_tform_target.z
    ]

    # Compute the unit vector.
    if np.linalg.norm(robot_rt_person_ewrt_vision) < 0.01:
        robot_rt_person_ewrt_vision_hat = vision_tform_robot.transform_point(1, 0, 0)
    else:
        robot_rt_person_ewrt_vision_hat = robot_rt_person_ewrt_vision / np.linalg.norm(
            robot_rt_person_ewrt_vision)

    # Starting at the person, back up meters along the unit vector.
    drop_position_rt_vision = [
        vision_tform_target.x + robot_rt_person_ewrt_vision_hat[0] * distance_margin,
        vision_tform_target.y + robot_rt_person_ewrt_vision_hat[1] * distance_margin,
        vision_tform_target.z + robot_rt_person_ewrt_vision_hat[2] * distance_margin
    ]

    # We also want to compute a rotation (yaw) so that we will face the person when dropping.
    # We'll do this by computing a rotation matrix with X along
    #   -robot_rt_person_ewrt_vision_hat (pointing from the robot to the person) and Z straight up:
    xhat = -robot_rt_person_ewrt_vision_hat
    zhat = [0.0, 0.0, 1.0]
    yhat = np.cross(zhat, xhat)
    mat = np.matrix([xhat, yhat, zhat]).transpose()
    heading_rt_vision = math_helpers.Quat.from_matrix(mat).to_yaw()

    return drop_position_rt_vision, heading_rt_vision

def get_walking_params(max_linear_vel, max_rotation_vel):
    max_vel_linear = geometry_pb2.Vec2(x=max_linear_vel, y=max_linear_vel)
    max_vel_se2 = geometry_pb2.SE2Velocity(linear=max_vel_linear, angular=max_rotation_vel)
    vel_limit = geometry_pb2.SE2VelocityLimit(max_vel=max_vel_se2)
    params = RobotCommandBuilder.mobility_params()
    params.vel_limit.CopyFrom(vel_limit)
    return params

def pose_dist(pose1, pose2):
    diff_vec = [pose1.x - pose2.x, pose1.y - pose2.y, pose1.z - pose2.z]
    return np.linalg.norm(diff_vec)

def wait_until_grasp_state_updates(grasp_override_command, robot_state_client):
    updated = False
    has_grasp_override = grasp_override_command.HasField("api_grasp_override")
    has_carry_state_override = grasp_override_command.HasField("carry_state_override")

    while not updated:
        robot_state = robot_state_client.get_robot_state()

        grasp_state_updated = (robot_state.manipulator_state.is_gripper_holding_item and
                               (grasp_override_command.api_grasp_override.override_request
                                == manipulation_api_pb2.ApiGraspOverride.OVERRIDE_HOLDING)) or (
                                    not robot_state.manipulator_state.is_gripper_holding_item and
                                    grasp_override_command.api_grasp_override.override_request
                                    == manipulation_api_pb2.ApiGraspOverride.OVERRIDE_NOT_HOLDING)
        carry_state_updated = has_carry_state_override and (
            robot_state.manipulator_state.carry_state
            == grasp_override_command.carry_state_override.override_request)
        updated = (not has_grasp_override or
                   grasp_state_updated) and (not has_carry_state_override or carry_state_updated)
        time.sleep(0.1)


if __name__ == "__main__":
    load_dotenv()
    #==================================Parse args===================================================
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument('-s', '--ml-service',
                        help='Service name of external machine learning server.', required=True)
    parser.add_argument('-m', '--model', help='Model name running on the external server.',
                        required=True)
    parser.add_argument('-c', '--confidence-piece',
                        help='Minimum confidence to return an object for the dogoy (0.0 to 1.0)',
                        default=0.5, type=float)
    parser.add_argument('-d', '--distance-margin', default=.5,
                        help='Distance [meters] that the robot should stop from the fiducial.')
    parser.add_argument('--limit-speed', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='If the robot should limit its maximum speed.')
    parser.add_argument('--avoid-obstacles', default=False, type=lambda x:
                        (str(x).lower() == 'true'),
                        help='If the robot should have obstacle avoidance enabled.')
    
    options = parser.parse_args()
    
    sdk = bosdyn.client.create_standard_sdk('TicTacSPOT')
    sdk.register_service_client(NetworkComputeBridgeClient)
    robot = sdk.create_robot(options.hostname)
    
    
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()

    network_compute_client = robot.ensure_client(NetworkComputeBridgeClient.default_service_name)
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    command_client = robot.ensure_client(RobotCommandClient.default_service_name)
    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    manipulation_api_client = robot.ensure_client(ManipulationApiClient.default_service_name)
    _world_object_client = robot.ensure_client(WorldObjectClient.default_service_name)
    
    lease_client.take()
    with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        pick_up(options, robot)
        