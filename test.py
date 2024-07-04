import asyncio
import subprocess
import time
from threading import Thread

import RPi.GPIO as GPIO

# 初始化上下左右角度为90度
ServoLeftRightPos = 20
ServoUpDownPos = 90
g_frontServoPos = 90
g_nowfrontPos = 0

# 小车电机引脚定义
IN1 = 20
IN2 = 21
IN3 = 19
IN4 = 26
ENA = 16
ENB = 13

# 超声波引脚定义
EchoPin = 0
TrigPin = 1

# 舵机引脚定义
FrontServoPin = 23
ServoUpDownPin = 9
ServoLeftRightPin = 11

# 红外避障引脚定义
AvoidSensorLeft = 12
AvoidSensorRight = 17

# 蜂鸣器引脚定义
buzzer = 8

# 灭火电机引脚设置
OutfirePin = 2
LED_R = 22
LED_G = 27
LED_B = 24
# 循迹红外引脚定义
# TrackSensorLeftPin1 TrackSensorLeftPin2 TrackSensorRightPin1 TrackSensorRightPin2
#      3                 5                  4                   18
TrackSensorLeftPin1 = 3  # 定义左边第一个循迹红外传感器引脚为3口
TrackSensorLeftPin2 = 5  # 定义左边第二个循迹红外传感器引脚为5口
TrackSensorRightPin1 = 4  # 定义右边第一个循迹红外传感器引脚为4口
TrackSensorRightPin2 = 18  # 定义右边第二个循迹红外传感器引脚为18口
# 光敏电阻
LdrSensorLeft = 17
LdrSensorRight = 12

# 小车速度变量
CarSpeedControl = 20
# 寻迹，避障，寻光变量
infrared_track_value = ''
infrared_avoid_value = ''
LDR_value = ''
g_lednum = 0
ServoPin = 23



GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
mark = True
mark2 = False
Finalturn = ''
IsContro = False


def init():
    global pwm_ENA
    global pwm_ENB
    global pwm_servo
    global pwm_LeftRightServo
    GPIO.setup(ENA, GPIO.OUT, initial=GPIO.HIGH)
    GPIO.setup(IN1, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(IN2, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(ENB, GPIO.OUT, initial=GPIO.HIGH)
    GPIO.setup(IN3, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(IN4, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(buzzer, GPIO.OUT, initial=GPIO.HIGH)
    GPIO.setup(OutfirePin, GPIO.OUT, initial=GPIO.HIGH)
    GPIO.setup(EchoPin, GPIO.IN)
    GPIO.setup(TrigPin, GPIO.OUT)
    GPIO.setup(ServoLeftRightPin, GPIO.OUT)
    GPIO.setup(ServoPin, GPIO.OUT)
    GPIO.setup(AvoidSensorLeft, GPIO.IN)
    GPIO.setup(AvoidSensorRight, GPIO.IN)
    GPIO.setup(LdrSensorLeft, GPIO.IN)
    GPIO.setup(LdrSensorRight, GPIO.IN)
    GPIO.setup(TrackSensorLeftPin1, GPIO.IN)
    GPIO.setup(TrackSensorLeftPin2, GPIO.IN)
    GPIO.setup(TrackSensorRightPin1, GPIO.IN)
    GPIO.setup(TrackSensorRightPin2, GPIO.IN)
    pwm_ENA = GPIO.PWM(ENA, 2000)
    pwm_ENB = GPIO.PWM(ENB, 2000)
    pwm_ENA.start(0)
    pwm_ENB.start(0)
    pwm_servo = GPIO.PWM(ServoPin, 50)
    pwm_servo.start(0)
    pwm_LeftRightServo = GPIO.PWM(ServoLeftRightPin, 50)
    pwm_LeftRightServo.start(0)
    pwm_LeftRightServo.ChangeDutyCycle(2.5 + 10 * 90 / 180)
    time.sleep(0.1)
    pwm_LeftRightServo.ChangeDutyCycle(0)
    GPIO.setup(LED_R, GPIO.OUT)
    GPIO.setup(LED_G, GPIO.OUT)
    GPIO.setup(LED_B, GPIO.OUT)
    GPIO.setup(LdrSensorLeft, GPIO.IN)
    GPIO.setup(LdrSensorRight, GPIO.IN)


def spin_left(delaytime):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    pwm_ENA.ChangeDutyCycle(80)
    pwm_ENB.ChangeDutyCycle(80)
    time.sleep(delaytime)


def spin_right():
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    pwm_ENA.ChangeDutyCycle(15)
    pwm_ENB.ChangeDutyCycle(15)


temp = 5.5


def servo_left(command):
    global temp
    try:
        command = int(command[4:])
        temp = temp + 10 * command / 180
        print('左转' + str(command))
        pwm_LeftRightServo.ChangeDutyCycle(temp)
        time.sleep(0.02)
        pwm_LeftRightServo.ChangeDutyCycle(0)
        time.sleep(0.02)
    except:
        pass


def Distance_test():
    num = 0
    ultrasonic = []
    while num < 5:
        distance = Distance()
        while int(distance) == -1:
            distance = Distance()
        while int(distance) >= 500 or int(distance) == 0:
            distance = Distance()
        ultrasonic.append(distance)
        num = num + 1
        time.sleep(0.01)
    print(ultrasonic)
    distance = (ultrasonic[1] + ultrasonic[2] + ultrasonic[3]) / 3
    return distance


def Distance():
    GPIO.output(TrigPin, GPIO.LOW)
    time.sleep(0.000002)
    GPIO.output(TrigPin, GPIO.HIGH)
    time.sleep(0.000015)
    GPIO.output(TrigPin, GPIO.LOW)
    t3 = time.time()
    while not GPIO.input(EchoPin):
        t4 = time.time()
        if (t4 - t3) > 0.03:
            return -1
    t1 = time.time()
    while GPIO.input(EchoPin):
        t5 = time.time()
        if (t5 - t1) > 0.03:
            return -1
    t2 = time.time()
    time.sleep(0.01)
    return ((t2 - t1) * 340 / 2) * 100


def servo_right(command):
    global temp
    try:
        command = int(command[5:])
        print('右转' + str(command))
        temp = temp - 10 * command / 180
        pwm_LeftRightServo.ChangeDutyCycle(temp)
        time.sleep(0.02)
        pwm_LeftRightServo.ChangeDutyCycle(0)
        time.sleep(0.04)
    except:
        pass


init()


async def shutdown():
    print("服务器关闭,不再处理后续相应")
    pwm_ENA.stop()
    pwm_ENB.stop()
    init()


def control_gpio():
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    pwm_ENA.ChangeDutyCycle(5)
    pwm_ENB.ChangeDutyCycle(5)


async def run():
    Server_data = "已收到指令前进,已经按你的命令做了."
    return Server_data


def stopBK():
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)


async def stop():
    Server_data = "已收到指令停止,已经按你的命令做了."
    return Server_data


HOST = '192.168.137.213'
PORT = 6666
current_angle = 90


async def set_servo_angle(angle):
    duty_cycle = 2.5 + 10 * angle / 180
    pwm_LeftRightServo.ChangeDutyCycle(duty_cycle)
    await asyncio.sleep(0.02)
    pwm_LeftRightServo.ChangeDutyCycle(0)


async def turn_left(offset: float):
    global current_angle
    new_angle = max(0, current_angle - offset)
    await set_servo_angle(new_angle)
    current_angle = new_angle
    print(f"已经左转 {offset} 度，当前角度 {new_angle}")


async def turn_right(offset: float):
    global current_angle
    new_angle = min(180, current_angle + offset)  # 确保角度不大于180
    await set_servo_angle(new_angle)  # 使用 await
    current_angle = new_angle
    print(f"已经右转 {offset} 度，当前角度 {new_angle}")


async def handle_client(reader, writer):
    while True:
        data = await reader.readuntil(separator=b'\n')
        message = data.decode().strip()
        if message:
            print(f"接收到指令: {message}")
            parts = message.split('/')
            if len(parts) == 2:
                command, param = parts
                try:
                    param = float(param)
                except ValueError:
                    print("参数格式错误")
                    return
                if command == "left":
                    await turn_left(param)
                elif command == "right":
                    await turn_right(param)
                else:
                    print("未知指令")
            else:
                print("命令格式错误")
            writer.write(f"已处理指令: {message}\n".encode())
            await writer.drain()
        else:
            break

    print("关闭连接")
    writer.close()
    await writer.wait_closed()


async def left(angle):
    print(f"向左转 {angle} 度")


async def right(angle):
    print(f"向右转 {angle} 度")


def monitor_gpio():
    last_sent = 0
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(12, GPIO.IN)
    try:
        while True:
            LdrSensorRightValue = GPIO.input(12)
            if LdrSensorRightValue == 0:
                print('发现红外异常')
                current_time = time.time()
                if current_time - last_sent > 0.3:
                    message = "9/n1"
                    try:
                        asyncio.run(send_data_to_server(message))
                    except Exception as e:
                        print("异常发生:", e)
                    last_sent = current_time
            time.sleep(0.1)
    except Exception as e:
        print("异常:", e)
    finally:
        GPIO.cleanup()


async def send_data_to_server(message):
    """ 异步发送数据到上位机 """
    try:
        reader, writer = await asyncio.open_connection(HOST, PORT)
        writer.write(message.encode())
        await writer.drain()
        writer.close()
        await writer.wait_closed()
    except Exception as e:
        print("发送失败:", e)


async def main():
    gpio_task = asyncio.create_task(monitor_gpio())
    server = await asyncio.start_server(
        handle_client, HOST, 8000)
    addr = server.sockets[0].getsockname()
    print(f'服务端启动，监听 {addr}')
    await gpio_task
    async with server:
        await server.serve_forever()


def start_mjpg_streamer():
    command = "./mjpg_streamer"
    input_plugin = "./input_uvc.so -f 60 -r 640x480"
    output_plugin = "./output_http.so -w ./www"
    cmd = [command, "-i", input_plugin, "-o", output_plugin]
    try:
        process = subprocess.Popen(cmd, cwd="/home/pi/SmartCar/mjpg-streamer/mjpg-streamer-experimental")
        print("mjpg_streamer 已启动")
        return process
    except Exception as e:
        print(f"启动 mjpg_streamer 时出错: {e}")
        return None


if __name__ == '__main__':
    process = start_mjpg_streamer()
    t1 = Thread(target=monitor_gpio)
    t1.start()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("进程即将停止,准备资源清理")
    t1.join()
