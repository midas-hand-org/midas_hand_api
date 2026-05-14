import serial
import time
import serial.tools.list_ports
from typing import List, Optional, Dict
import logging

# -------------------------- 设备协议配置（与GEN3传感器匹配）--------------------------
AUTO_PUSH_REG = 0x0017               # 自动回传控制寄存器（1=开启，0=关闭）
AUTO_PUSH_FRAME_HEAD = b"\xAA\x56"   # 自动回传数据帧头
VERSION_REG = 0x0000                 # 版本号寄存器地址
VERSION_DATA_LEN = 0x000F            # 版本号数据长度（15字节）
DATA_TYPE_REG = 0x0016               # 数据类型组合寄存器
BAUDRATE = 921600                    # 高速通信波特率
TIMEOUT_CMD = 1.0                    # 指令响应超时（秒）
TIMEOUT_AUTO_PUSH = 0.05             # 自动回传监听超时（秒）

# 帧结构常量
REQ_HEAD = b"\x55\xAA"               # 请求帧头（主机→传感器）
RESP_HEAD_GENERAL = b"\xAA\x55"      # 普通响应帧头（传感器→主机）
RESP_HEAD_AUTO_PUSH = b"\xAA\x56"    # 自动回传相关响应帧头
RESERVED = b"\x00"                   # 预留字段
FUNC_READ = 0x03                     # 读寄存器功能码
FUNC_WRITE = 0x10                    # 写寄存器功能码

# 指令示例（用于校验）
CMD_EXAMPLE_AUTO_PUSH = "55AA00101700010001D8"  # 开启自动推送示例指令
CMD_EXAMPLE_VERSION = "55AA000300000F00EF"      # 读版本号示例指令

# -------------------------- 日志配置 --------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("sensor_comm.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def send_hex_cmd(ser: serial.Serial, hex_cmd: str) -> bool:
    """发送16进制指令"""
    try:
        cmd_bytes = bytes.fromhex(hex_cmd)
        ser.flushOutput()
        sent_len = ser.write(cmd_bytes)
        if sent_len != len(cmd_bytes):
            logger.error(f"指令发送不完整：应发{len(cmd_bytes)}字节，实发{sent_len}字节 | 指令：{hex_cmd}")
            return False
        logger.debug(f"指令发送成功：{hex_cmd}（{sent_len}字节）")
        return True
    except (serial.SerialException, ValueError, Exception) as e:
        logger.error(f"发送错误：{str(e)} | 指令：{hex_cmd}", exc_info=True)
        return False


def read_serial_data(ser: serial.Serial, timeout: float = TIMEOUT_CMD, expected_head: Optional[bytes] = None) -> Optional[bytes]:
    """读取串口数据（支持指定预期帧头）"""
    try:
        start_time = time.time()
        recv_data = b""
        while time.time() - start_time < timeout:
            if ser.in_waiting > 0:
                chunk = ser.read(ser.in_waiting)
                recv_data += chunk
                logger.debug(f"接收数据片段：{chunk.hex()}（{len(chunk)}字节）")
                
                # 如果指定了预期帧头，检查是否已收到
                if expected_head and expected_head in recv_data:
                    # 从帧头开始截取数据
                    head_pos = recv_data.find(expected_head)
                    recv_data = recv_data[head_pos:]
                    break  # 找到预期帧头后退出等待
                    
                start_time = time.time()  # 重置超时计时器
                time.sleep(0.005)  # 等待可能的后续数据
            time.sleep(0.001)
        
        if recv_data:
            logger.debug(f"完整数据接收：{recv_data.hex()}（{len(recv_data)}字节）")
            return recv_data
        logger.warning(f"超时未接收数据（{timeout}秒）")
        return None
    except Exception as e:
        logger.error(f"读取错误：{str(e)}", exc_info=True)
        return None


def calc_lrc(data: bytes) -> int:
    """计算LRC校验（累加→取反→加1→低8位）"""
    try:
        lrc_sum = 0
        for byte in data:
            lrc_sum = (lrc_sum + byte) & 0xFF  # 8位累加防溢出
        lrc = ((~lrc_sum) + 1) & 0xFF         # 补码计算
        logger.debug(f"LRC计算：{data.hex()} → 0x{lrc:02X}")
        return lrc
    except Exception as e:
        logger.error(f"LRC计算错误：{str(e)}", exc_info=True)
        return 0


def build_request_frame(func_code: int, reg_addr: int, data_len: int, write_data: bytes = b"") -> Optional[str]:
    """构建请求帧（完全匹配协议：Head+预留+功能码+寄存器地址+数据长度+数据+LRC）"""
    try:
        # 寄存器地址和数据长度均采用小端模式（匹配协议要求）
        reg_addr_bytes = reg_addr.to_bytes(2, byteorder="little")
        data_len_bytes = data_len.to_bytes(2, byteorder="little")
        
        # 组装帧主体（不含LRC）
        frame_without_lrc = (
            REQ_HEAD + 
            RESERVED + 
            func_code.to_bytes(1, "big") + 
            reg_addr_bytes + 
            data_len_bytes + 
            write_data
        )
        
        # 计算LRC并补全帧
        lrc = calc_lrc(frame_without_lrc).to_bytes(1, "big")
        full_frame = frame_without_lrc + lrc
        frame_hex = full_frame.hex().upper()
        
        # 校验与示例指令一致性
        if func_code == FUNC_WRITE and reg_addr == AUTO_PUSH_REG and write_data == b"\x01" and data_len == 1:
            if frame_hex != CMD_EXAMPLE_AUTO_PUSH:
                logger.warning(f"开启自动推送指令与示例不符：生成{frame_hex}，示例{CMD_EXAMPLE_AUTO_PUSH}")
        if func_code == FUNC_READ and reg_addr == VERSION_REG and data_len == VERSION_DATA_LEN:
            if frame_hex != CMD_EXAMPLE_VERSION:
                logger.warning(f"读版本号指令与示例不符：生成{frame_hex}，示例{CMD_EXAMPLE_VERSION}")
        
        logger.debug(f"请求帧构建：{frame_hex}（{len(full_frame)}字节）")
        return frame_hex
    except Exception as e:
        logger.error(f"请求帧构建错误：{str(e)}", exc_info=True)
        return None


def parse_auto_response(response: bytes) -> Optional[Dict]:
    """解析自动回传相关响应（包括关闭指令响应），帧头为AA 56"""
    try:
        # 基础校验：最小帧长度
        if len(response) < 7:
            logger.error(f"自动回传响应帧过短：{len(response)}字节 | 数据：{response.hex()}")
            return None
        
        # 帧头校验（AA 56）
        if response[:2] != RESP_HEAD_AUTO_PUSH:
            logger.error(f"自动回传响应帧头错误：{response[:2].hex()} 期望{RESP_HEAD_AUTO_PUSH.hex()}")
            return None
        
        # 提取基础字段
        parsed = {
            "head": response[:2].hex(),
            "reserved": response[2],
            "valid_frame_len": int.from_bytes(response[3:5], "little"),  # 有效帧长度：小端
            "error_code": response[5],
            "valid_data": b"",
            "valid_data_len": 0,
            "lrc_valid": False,
            "lrc_calc": 0,
            "lrc_recv": response[-1] if len(response) >= 7 else 0
        }
        
        # 计算有效数据长度
        parsed["valid_data_len"] = parsed["valid_frame_len"] - 1
        
        # 提取有效数据
        if parsed["valid_data_len"] > 0:
            data_end_pos = 6 + parsed["valid_data_len"]
            if data_end_pos <= len(response) - 1:  # 预留LRC位置
                parsed["valid_data"] = response[6:data_end_pos]
            else:
                parsed["valid_data"] = response[6:-1]  # 截取到LRC前
                logger.warning(f"自动回传响应数据不完整：期望{parsed['valid_data_len']}字节，实际{len(parsed['valid_data'])}字节")
        
        # LRC校验
        if len(response) >= 6 + parsed["valid_data_len"] + 1:
            parsed["lrc_calc"] = calc_lrc(response[:-1])
            parsed["lrc_valid"] = (parsed["lrc_calc"] == parsed["lrc_recv"])
            if not parsed["lrc_valid"]:
                logger.warning(f"自动回传响应LRC校验失败：计算0x{parsed['lrc_calc']:02X}，实际0x{parsed['lrc_recv']:02X}")
        
        logger.debug(f"自动回传响应解析完成：{parsed}")
        return parsed
    except Exception as e:
        logger.error(f"自动回传响应解析错误：{str(e)} | 数据：{response.hex()}", exc_info=True)
        return None


def parse_response(response: bytes) -> Optional[Dict]:
    """解析普通设备响应帧（Head+预留+功能码+地址+数据长度+数据+LRC）"""
    try:
        # 基础校验：最小帧长度
        if len(response) < 8:  # 无数据时最小8字节
            logger.error(f"响应帧过短：{len(response)}字节 | 数据：{response.hex()}")
            return None
        
        # 帧头校验（普通响应为AA 55）
        if response[:2] != RESP_HEAD_GENERAL:
            logger.error(f"帧头不匹配：{response[:2].hex()} 期望{RESP_HEAD_GENERAL.hex()} | 数据：{response.hex()}")
            return None
        
        # 提取字段（所有多字节字段采用小端解析）
        parsed = {
            "is_error": False,
            "reserved": response[2],
            "func_code": response[3],
            "reg_addr": int.from_bytes(response[4:6], "little"),
            "data_len": int.from_bytes(response[6:8], "little"),
            "actual_data_len": len(response) - 9 if len(response) > 8 else 0,
            "data": b"",
            "lrc_valid": False,
            "lrc_calc": 0,
            "lrc_recv": response[-1] if len(response) >= 9 else 0
        }
        
        # 错误响应处理（功能码最高位为1表示错误）
        if (parsed["func_code"] & 0x80) != 0:
            parsed["is_error"] = True
            parsed["error_code"] = parsed["func_code"] & 0x7F
            logger.warning(f"设备错误：0x{parsed['error_code']:02X} | 地址0x{parsed['reg_addr']:04X}")
            return parsed
        
        # 功能码校验
        if parsed["func_code"] not in [FUNC_READ, FUNC_WRITE]:
            logger.error(f"无效功能码：0x{parsed['func_code']:02X}")
            return None
        
        # 提取有效数据
        if parsed["data_len"] > 0 and len(response) >= 8 + parsed["data_len"] + 1:
            parsed["data"] = response[8:8 + parsed["data_len"]]
            logger.debug(f"提取数据：{parsed['data'].hex()}（{len(parsed['data'])}字节）")
        elif parsed["data_len"] > 0:
            parsed["data"] = response[8:-1] if len(response) > 8 else b""
            logger.warning(f"数据不完整：期望{parsed['data_len']}字节，实际{len(parsed['data'])}字节")
        
        # LRC校验
        if len(response) >= 9:
            parsed["lrc_calc"] = calc_lrc(response[:-1])
            parsed["lrc_valid"] = (parsed["lrc_calc"] == parsed["lrc_recv"])
            if not parsed["lrc_valid"]:
                logger.warning(f"LRC校验失败：计算0x{parsed['lrc_calc']:02X}，实际0x{parsed['lrc_recv']:02X}")
        
        # 数据长度一致性校验
        if parsed["data_len"] != len(parsed["data"]):
            logger.warning(f"数据长度不匹配：期望{parsed['data_len']}，实际{len(parsed['data'])}")
        
        return parsed
    except Exception as e:
        logger.error(f"响应解析错误：{str(e)} | 数据：{response.hex()}", exc_info=True)
        return None


def read_register(ser: serial.Serial, reg_addr: int, read_len: int) -> Optional[bytes]:
    """读取寄存器数据（功能码0x03）"""
    if not (1 <= read_len <= 512):
        logger.error(f"无效读取长度：{read_len}字节（协议限制1-512字节）")
        return None
    
    read_frame = build_request_frame(FUNC_READ, reg_addr, read_len)
    if not read_frame:
        logger.error(f"构建读请求失败：0x{reg_addr:04X}，{read_len}字节")
        return None
    
    if not send_hex_cmd(ser, read_frame):
        logger.error(f"发送读请求失败")
        return None
    
    time.sleep(0.2)  # 预留响应时间
    response = read_serial_data(ser, TIMEOUT_CMD, RESP_HEAD_GENERAL)
    if not response:
        logger.error(f"未收到读响应")
        return None
    
    parsed = parse_response(response)
    if not parsed or parsed["is_error"] or parsed["func_code"] != FUNC_READ:
        logger.error(f"读操作失败")
        return None
    
    return parsed["data"]


def write_register(ser: serial.Serial, reg_addr: int, write_data: bytes, is_auto_push: bool = False) -> bool:
    """写入寄存器数据（功能码0x10），is_auto_push标识是否为自动回传相关操作"""
    write_len = len(write_data)
    if not (1 <= write_len <= 10):
        logger.error(f"无效写入长度：{write_len}字节（协议限制1-10字节）")
        return False
    
    write_frame = build_request_frame(FUNC_WRITE, reg_addr, write_len, write_data)
    if not write_frame:
        logger.error(f"构建写请求失败：0x{reg_addr:04X}")
        return False
    
    if not send_hex_cmd(ser, write_frame):
        logger.error(f"发送写请求失败")
        return False
    
    time.sleep(0.2)  # 预留响应时间
    
    # 根据是否为自动回传相关操作选择不同的帧头
    expected_head = RESP_HEAD_AUTO_PUSH if is_auto_push else RESP_HEAD_GENERAL
    response = read_serial_data(ser, TIMEOUT_CMD, expected_head)
    if not response:
        logger.error(f"未收到写响应")
        return False
    
    # 解析响应
    if is_auto_push:
        parsed = parse_auto_response(response)
        # 自动回传响应通过error_code判断是否成功（0表示成功）
        if not parsed or parsed["error_code"] != 0:
            logger.error(f"自动回传相关写操作失败，错误码：0x{parsed['error_code']:02X}" if parsed else "自动回传相关写操作响应解析失败")
            return False
    else:
        parsed = parse_response(response)
        if not parsed or parsed["is_error"] or parsed["func_code"] != FUNC_WRITE:
            logger.error(f"写操作失败")
            return False
        
        # 验证写入状态（返回数据为0表示成功）
        if len(parsed["data"]) > 0:
            write_status = int.from_bytes(parsed["data"], "little")
            if write_status != 0:
                logger.error(f"写入状态错误：0x{write_status:02X}（0表示成功）")
                return False
    
    return True


def disable_auto_push(ser: serial.Serial) -> bool:
    """关闭自动回传功能（写入0x0017寄存器为0x00）"""
    try:
        # 直接构建并发送关闭指令，不等待响应
        disable_cmd = build_request_frame(FUNC_WRITE, AUTO_PUSH_REG, 1, b"\x00")
        if not disable_cmd:
            logger.error("构建关闭自动回传指令失败")
            return False
            
        # 发送指令但不验证响应
        if send_hex_cmd(ser, disable_cmd):
            logger.info("关闭自动回传指令已发送")
            # 短暂延迟确保指令被接收
            time.sleep(0.1)
            return True
        return False
    except Exception as e:
        logger.error(f"关闭自动回传失败：{str(e)}", exc_info=True)
        return False


def enable_auto_push(ser: serial.Serial) -> bool:
    """开启自动回传功能（写入0x0017寄存器为0x01）"""
    return write_register(ser, AUTO_PUSH_REG, b"\x01", is_auto_push=True)

def parse_auto_push_data(data: bytes, expected_length: int = 0) -> Optional[Dict]:
    """解析自动回传数据帧（AA56头+预留+有效帧长度+总错误码+有效数据+LRC）"""
    try:
        # 基础校验：最小帧长度（Head2+预留1+有效帧长度2+总错误码1+LRC1=7字节）
        if len(data) < 7:
            logger.error(f"自动回传帧过短：{len(data)}字节 | 数据：{data.hex()}")
            return None
        
        # 帧头校验
        if data[:2] != AUTO_PUSH_FRAME_HEAD:
            logger.error(f"自动回传帧头错误：{data[:2].hex()} 期望{AUTO_PUSH_FRAME_HEAD.hex()} | 数据：{data.hex()}")
            return None
        
        # 提取基础字段
        parsed = {
            "head": data[:2].hex(),
            "reserved": data[2],
            "valid_frame_len": int.from_bytes(data[3:5], "little"),  # 有效帧长度：小端
            "error_code": data[5],
            "valid_data": b"",
            "valid_data_len": 0,
            "expected_data_len": expected_length,
            "length_match": False,
            "lrc_valid": False,
            "lrc_calc": 0,
            "lrc_recv": data[-1] if len(data) >= 7 else 0
        }
        
        # 计算有效数据长度（有效帧长度 = 有效数据长度 + 1）
        parsed["valid_data_len"] = parsed["valid_frame_len"] - 1
        
        # 提取有效数据
        if parsed["valid_data_len"] > 0:
            data_end_pos = 6 + parsed["valid_data_len"]
            if data_end_pos <= len(data) - 1:  # 预留LRC位置
                parsed["valid_data"] = data[6:data_end_pos]
            else:
                parsed["valid_data"] = data[6:-1]  # 截取到LRC前
                logger.warning(f"自动回传数据不完整：期望{parsed['valid_data_len']}字节，实际{len(parsed['valid_data'])}字节")
        
        # 校验数据长度是否符合预期配置
        if expected_length > 0:
            parsed["length_match"] = (parsed["valid_data_len"] == expected_length)
            if not parsed["length_match"]:
                logger.warning(f"数据长度与配置不符：实际{parsed['valid_data_len']}字节，期望{expected_length}字节")
        
        # LRC校验
        if len(data) >= 6 + parsed["valid_data_len"] + 1:
            parsed["lrc_calc"] = calc_lrc(data[:-1])
            parsed["lrc_valid"] = (parsed["lrc_calc"] == parsed["lrc_recv"])
            if not parsed["lrc_valid"]:
                logger.warning(f"自动回传LRC校验失败：计算0x{parsed['lrc_calc']:02X}，实际0x{parsed['lrc_recv']:02X}")
        
        return parsed
    except Exception as e:
        logger.error(f"自动回传解析错误：{str(e)} | 数据：{data.hex()}", exc_info=True)
        return None


def get_device_version(ser: serial.Serial) -> Optional[str]:
    """获取设备版本号"""
    version_data = read_register(ser, VERSION_REG, VERSION_DATA_LEN)
    if version_data:
        try:
            ascii_version = version_data.decode("ascii", errors="ignore").strip()
            return f"ASCII: {ascii_version} | 16进制: {version_data.hex().upper()}"
        except:
            return f"16进制: {version_data.hex().upper()}"
    return None


def monitor_auto_push(ser: serial.Serial, duration: Optional[float] = None) -> None:
    """监听自动回传数据（不使用模组配置）"""
    start_time = time.time()
    logger.info(f"开始监听自动回传")
    print("\n===== 开始接收自动回传数据 =====")
    print(f"按Ctrl+C停止")
    print("-" * 60)
    
    try:
        while True:
            # 检查监听时长
            if duration and time.time() - start_time > duration:
                logger.info(f"监听超时（{duration}秒）")
                break
            
            # 读取自动回传数据
            push_data = read_serial_data(ser, TIMEOUT_AUTO_PUSH, AUTO_PUSH_FRAME_HEAD)
            if push_data:
                parsed = parse_auto_push_data(push_data)
                if parsed:
                    print(f"[{time.strftime('%H:%M:%S')}]")
                    print(f"  帧头: {parsed['head']} | 错误码: 0x{parsed['error_code']:02X}")
                    print(f"  数据长度: {parsed['valid_data_len']}字节" )
                    print(f"  数据内容: {parsed['valid_data'].hex().upper()}")
                    print(f"  LRC校验: {'通过' if parsed['lrc_valid'] else '失败'}")
                    print("-" * 60)
    except KeyboardInterrupt:
        logger.info("用户中断监听")
        print("\n用户手动停止监听")
    except Exception as e:
        logger.error(f"监听错误：{str(e)}", exc_info=True)
        print(f"\n监听错误：{str(e)}")


def main():
    logger.info("=== GEN3触觉传感器程序启动 ===")
    
    # 扫描可用串口
    available_ports = list(serial.tools.list_ports.comports())
    if not available_ports:
        logger.error("无可用串口")
        print("错误：未找到可用串口")
        return
    
    # 选择串口
    print("\n可用串口：")
    for i, port in enumerate(available_ports, 1):
        print(f"  {i}. {port.device} - {port.description}")
    
    try:
        choice = int(input(f"\n选择串口（1-{len(available_ports)}）: "))
        selected_port = available_ports[choice - 1].device
    except (ValueError, IndexError):
        print("输入无效，选择第一个串口")
        selected_port = available_ports[0].device
    
    # 初始化串口
    ser = None
    try:
        ser = serial.Serial(
            port=selected_port,
            baudrate=BAUDRATE,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=1,
            write_timeout=0.5,
            inter_byte_timeout=0.001  # 高速通信防字节丢失
        )
        
        if not ser.is_open:
            ser.open()
        print(f"\n连接串口：{ser.name}（波特率{BAUDRATE}）")
        
        # 发送版本号指令
        print("\n发送版本号指令...")
        version_cmd = "55AA000300000F00EF"
        if send_hex_cmd(ser, version_cmd):
            print("版本号指令发送成功，等待响应...")
            response = read_serial_data(ser, TIMEOUT_CMD, RESP_HEAD_GENERAL)
            if response:
                print(f"收到版本号响应：{response.hex().upper()}")
                # 尝试解析版本号
                parsed = parse_response(response)
                if parsed and not parsed["is_error"]:
                    try:
                        ascii_version = parsed["data"].decode("ascii", errors="ignore").strip()
                        print(f"解析版本号：ASCII: {ascii_version}")
                    except:
                        print(f"解析版本号：16进制: {parsed['data'].hex().upper()}")
            else:
                print("未收到版本号响应")
        else:
            print("版本号指令发送失败")

        time.sleep(1)  

        # 发送开启自动回传指令
        print("\n发送开启自动回传指令...")
        auto_push_cmd = "55AA00101700010001D8"
        if send_hex_cmd(ser, auto_push_cmd):
            print("开启自动回传指令发送成功，等待响应...")
            response = read_serial_data(ser, TIMEOUT_CMD, RESP_HEAD_AUTO_PUSH)
            if response:
                print(f"收到自动回传响应：{response.hex().upper()}")
                # 尝试解析响应
                parsed = parse_auto_response(response)
                if parsed and parsed["error_code"] == 0:
                    print("自动回传已成功开启")
                    # 直接回传
                    monitor_auto_push(ser)
                else:
                    print(f"自动回传开启失败，错误码：0x{parsed['error_code']:02X}" if parsed else "自动回传开启失败")
            else:
                print("未收到自动回传指令响应")
        else:
            print("开启自动回传指令发送失败")
    
        time.sleep(0.1) 
    
    except serial.SerialException as e:
        logger.error(f"串口错误：{str(e)}")
        print(f"\n串口错误：{str(e)}")
    except KeyboardInterrupt:
        print("\n用户中断程序")
    except Exception as e:
        logger.error(f"程序错误：{str(e)}", exc_info=True)
        print(f"\n程序错误：{str(e)}")
    finally:
        if ser and ser.is_open:
            print("\n发送关闭自动回传指令...")
            disable_success = disable_auto_push(ser)
            if disable_success:
                print("关闭自动回传指令已发送")
            else:
                print("关闭自动回传指令发送失败")
            ser.close()
            print("串口已关闭")
    
    logger.info("=== 程序结束 ===")


if __name__ == "__main__":
    main()
    