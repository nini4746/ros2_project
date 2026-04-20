num1 = float(input("첫 번쨰 숫자를 입력하세요 : "))
num2 = float(input("첫 번쨰 숫자를 입력하세요 : "))

op = input("연산자(+, -, *, /)를 선택하세요 : ")

if op == '+':
    print(f"결과: {num1 + num2}")
elif op == '-':
    print(f"결과: {num1 - num2}")
elif op == '*':
    print(f"결과: {num1 * num2}")
elif op == '/':
    print(f"결과: {num1 / num2}")
else:
    print("잘못된 연산자입니다.") # 예외 상황 처리



# result = num1 + num2

# print(f"결과 {num1} + {num2} = {result}")

