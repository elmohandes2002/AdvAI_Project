from controller import Robot

robot = Robot()
timestep = int(robot.getBasicTimeStep())

left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')

left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

receiver = robot.getDevice('receiver')
receiver.enable(timestep)

while robot.step(timestep) != -1:
    if receiver.getQueueLength() > 0:
        message = receiver.getString()
        linear_vel, angular_vel = map(float, message.split(','))

        # Very simple differential drive control
        left_speed = linear_vel - angular_vel
        right_speed = linear_vel + angular_vel

        left_motor.setVelocity(left_speed)
        right_motor.setVelocity(right_speed)

        receiver.nextPacket()
