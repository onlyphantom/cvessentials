digits = ['a', 'b', 'c', 'd']

contracts = {
    # salesperson: contract value, duration
    'adam':(500, 2),
    'brian':(300, 1.5),
    'canny':(1000, 4)
}

# for i in range(len(digits)):
#     print(i, digits[i])
# better written as:
for i, d in enumerate(digits):
    print(i, d)

print('---')
print(dict(enumerate(digits)))

for i, c in enumerate(contracts):
    print(i, c)

for i, v in enumerate(contracts.values()):
    print(i, v)

d = {i+1:(k,f'${v1} for {v2} years') for i, (k,(v1, v2)) in enumerate(contracts.items())}
print(d)