from controller import Robot

robot = Robot()
timestep = int(robot.getBasicTimeStep())

left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')

# ps = [robot.getDevice(f'ps{i}') for i in range(8)]
# for sensor in ps:
    # sensor.enable(timestep)

left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

receiver = robot.getDevice('receiver')
receiver.enable(timestep)
receiver.setChannel(1)

robot_name = robot.getName()  # e.g., "CLEANING_ROBOT_2"
robot_id = int(robot_name.split('_')[-1])  # gets 2

# emitter = robot.getDevice('emitter')
# emitter.setChannel(10 + robot_id)


while robot.step(timestep) != -1:
    if receiver.getQueueLength() > 0:
        message = receiver.getString()
        receiver.nextPacket()
        # Expected format: "robot_id:linear_vel,angular_vel"

        robot_id, command = message.split(':')
        parts = list(map(float, command.split(',')))
        linear_vel, angular_vel = parts[0], parts[1]

        # Basic differential drive
        left_speed = linear_vel - angular_vel
        right_speed = linear_vel + angular_vel
        left_motor.setVelocity(left_speed)
        right_motor.setVelocity(right_speed)

        # Read sensor values and send to supervisor
        # sensor_values = [str(sensor.getValue()) for sensor in ps]
        # sensor_payload = ','.join(sensor_values)


        Send: "robot_id:val0,val1,...val7"
        # emitter.send(f"{robot_id}:{sensor_payload}")
