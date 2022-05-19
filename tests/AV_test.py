import gtab
my_path = "../GTAB_banks"
t = gtab.GTAB(dir_path=my_path)

for i in range(0, 11):
  # Run for Jan to Jun of one year
  timeframe_str = str(2010+i) + "-01-01 " + str(2010+i) + "-07-01"
  try:
    t.set_active_gtab("google_anchorbank_geo=US_timeframe=" + timeframe_str + ".tsv")
  except:
    print('Anchorbank %s not found'%(timeframe_str))
  # Run for Jul to Dec of one year
  timeframe_str = str(2010+i) + "-07-01 " + str(2011+i) + "-01-01"
  try:
    t.set_active_gtab("google_anchorbank_geo=US_timeframe=" + timeframe_str + ".tsv")
  except:
    print('Anchorbank %s not found'%(timeframe_str))
