# Aerial
for num in {1..10}
do  
mkdir block_$num/input
tar -xvf /media/allen/新加卷1/data_for_citygaussian2/MatrixCity/aerial/train/block_$num.tar
mv block_$num/*.png block_$num/input
done

# Street
#mkdir small_city_road_down/input
#tar -xvf small_city_road_down.tar
#mv small_city_road_down/*.png small_city_road_down/input

#mkdir small_city_road_horizon/input
#tar -xvf small_city_road_horizon.tar
#mv small_city_road_horizon/*.png small_city_road_horizon/input

#mkdir small_city_road_outside/input
#tar -xvf small_city_road_outside.tar
#mv small_city_road_outside/*.png small_city_road_outside/input

#mkdir small_city_road_vertical/input
#tar -xvf small_city_road_vertical.tar
#mv small_city_road_vertical/*.png small_city_road_vertical/input
