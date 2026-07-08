#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import smbus2
import time

I2C_BUS = 7
SHT40_ADDR = 0x44
CMD_MEASURE_HIGH_PRECISION = 0xFD

class SHT40:
    def __init__(self, bus_num=I2C_BUS, addr=SHT40_ADDR):
        self.bus = smbus2.SMBus(bus_num)
        self.addr = addr

    def _crc8(self, data):
        crc = 0xFF
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 0x80:
                    crc = (crc << 1) ^ 0x31
                else:
                    crc = crc << 1
        return crc & 0xFF

    def read_temp_humidity(self):
        try:
            write_msg = smbus2.i2c_msg.write(self.addr, [CMD_MEASURE_HIGH_PRECISION])
            self.bus.i2c_rdwr(write_msg)
            time.sleep(0.01)

            read_msg = smbus2.i2c_msg.read(self.addr, 6)
            self.bus.i2c_rdwr(read_msg)
            data = list(read_msg)

            temp_raw = (data[0] << 8) | data[1]
            temp_c = -45.0 + 175.0 * temp_raw / 65535.0

            hum_raw = (data[3] << 8) | data[4]
            hum_rh = -6.0 + 125.0 * hum_raw / 65535.0
            hum_rh = max(0.0, min(100.0, hum_rh))

            return temp_c, hum_rh
        except Exception as e:
            print(f"[错误] {e}")
            return None

    def close(self):
        self.bus.close()

if __name__ == '__main__':
    print("SHT40 温湿度传感器测试\n" + "=" * 40)
    sensor = SHT40()
    print("开始读取温湿度数据 (Ctrl+C 停止)\n")
    try:
        while True:
            result = sensor.read_temp_humidity()
            if result:
                temp, hum = result
                print(f"温度: {temp:6.2f}°C | 湿度: {hum:6.2f}%")
            else:
                print("读取失败")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n测试结束")
    finally:
        sensor.close()
