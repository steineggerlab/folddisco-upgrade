#Script to show statistics of domain length for selected pdbs

INFO1=result/domain-list-pdbexists.txt
INFO2=result/domain-list-selected.txt
OUTPUT=result/selected_with_length.txt
STATS=result/domain-length-list.txt 

#Gathering domain length
awk '
  BEGIN { OFS = "\t" }   
  FNR==NR && !/^#/ { len[$1]=$11; next }     # 1st pass: map[domain_id] = length
  !/^#/ {
    id=$1
    L = (id in len) ? len[id] : "NA"
    print $0, L                               # 기존 1~4열 뒤에 길이 추가
  }
' "$INFO1" "$INFO2"  > "$OUTPUT"

awk '!/^#/ {print $6}' "$OUTPUT" > "$STATS"

#Domain length statistics and plots
awk -v BINS=30 '
{
  a[++N]=$1
  if(N==1 || $1<min) min=$1
  if(N==1 || $1>max) max=$1
}
END {
  if(N==0) {
    print "NA"
    exit
  }
  width=(max-min)/BINS
  if(width<=0) width=1
  for(i=1;i<=N;i++){
    idx=int((a[i]-min)/width)
    if(idx>=BINS) idx=BINS-1
    c[idx]++
  }
  for(i=0;i<BINS;i++){
    lo=min+i*width; hi=lo+width
    n=(c[i]?c[i]:0)
    printf "%6.0f - %6.0f | %6d ", lo, hi, n
    print ""
  }
}' "$STATS"

awk '
{
  x[NR]=$1; s+=$1
  if(NR==1 || $1<min) min=$1
  if(NR==1 || $1>max) max=$1
}
END {
  n=NR
  mean = s/n
  asort(x)   # 배열 정렬

  # median
  if(n%2==1) median = x[(n+1)/2]
  else       median = (x[n/2] + x[n/2+1])/2

  # quantiles
  q1 = x[int(n*0.25+0.5)]
  q3 = x[int(n*0.75+0.5)]

  print "count="n
  print "mean="mean
  print "median="median
  print "min="min
  print "q1="q1
  print "q3="q3
  print "max="max
}' "$STATS"