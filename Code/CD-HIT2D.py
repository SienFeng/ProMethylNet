import subprocess
import os

def windows_path_to_wsl_path(win_path: str) -> str:
    # 使用 os.path.splitdrive 分解盘符
    drive, tail = os.path.splitdrive(win_path)
    if drive:
        drive_letter = drive[0].lower()
    else:
        drive_letter = ''
    # 将反斜杠替换为斜杠，并确保 tail 以 "/" 开头
    tail = tail.replace("\\", "/")
    if not tail.startswith("/"):
        tail = "/" + tail
    return f"/mnt/{drive_letter}{tail}"

def run_cd_hit_with_wsl(input_win_path, output_win_path, identity=0.4, threads=30, memory=51200, cd_hit_exe="cd-hit"):
    # 将 Windows 路径转换为 WSL 路径
    wsl_input = windows_path_to_wsl_path(input_win_path)
    wsl_output = windows_path_to_wsl_path(output_win_path)

    # 构造 cd-hit 命令行参数，在 WSL 下执行
    cmd = [
        "wsl",              # 使用 WSL 来执行Linux命令
        cd_hit_exe,         # cd-hit命令
        "-i", wsl_input,
        "-o", wsl_output,
        "-c", str(identity),
        "-n", "2",          # 针对 0.9 ~ 1.0 同一性阈值时，推荐使用 -n 5
        "-T", str(threads),
        "-M", str(memory)
    ]

    print("即将在 WSL 中执行命令：", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
        print(f"去冗余完成。输出文件(WSL路径): {wsl_output}")
        print(f"对应Windows路径: {output_win_path}")
    except subprocess.CalledProcessError as e:
        print("运行 cd-hit 时出错：", e)

if __name__ == "__main__":
    # 修改为你的实际文件路径，注意使用原始字符串 (在字符串前加 r) 避免转义问题。
    input_fasta_win = r"E:\GraduationProject\MethylationDatabase\dbPTM\dbPTM_sprot_with_methylation.fasta"
    output_fasta_win = r"E:\GraduationProject\MethylationDatabase\dbPTM\dbPTM_sprot_with_methylation_deduplicated1.fasta"

    run_cd_hit_with_wsl(
        input_win_path=input_fasta_win,
        output_win_path=output_fasta_win,
        identity=0.4,
        threads=30,
        memory=51200
    )