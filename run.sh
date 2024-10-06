env_name=$1
for i in 1
do
  date_str=`date +%Y.%m.%d_%H.%M.%S`
  echo " program start time : " + $date_str
  nohup python -u main.py --config=config.${env_name} > /dev/null 2>error.log &
  sleep 30
done